from scipy.spatial.transform import Rotation as R

from utils.io_exr import read_exr
from datasets.base_datasets import *


class PanoDataset(BaseDataset):
	"""Pano Dataset."""

	def __init__(
		self, 
		data_dir, 
		split='train', 
		white_bkgd=False, 
		batch_type='all_images',
		factor=0, 
		num=None, 
		num_start=0, 
		range=(0, 10), 
		normalize_depth=False, 
		num_per_epoch=512, 
		origin=None, 
		scale=None, 
		rot=None, 
		reform_cam=False, 
		meta_file='transforms_all',
	):
		super(PanoDataset, self).__init__(data_dir, split, white_bkgd, batch_type, factor)

		self.num = num
		self.num_start = num_start
		self.near = range[0]
		self.far = range[1]
		self.normalize_depth = normalize_depth
		self.num_per_epoch = num_per_epoch

		self.origin = origin
		self.scale = scale
		self.rot = rot
		self.reform_cam = reform_cam
		self.meta_file = meta_file

		if split == 'train':
			self._train_init()
		else:
			assert batch_type == 'single_image', 'The batch_type can only be single_image without flatten'
			self._val_init()
	
	def _load_renderings(self):
		"""Load images from disk."""
		with open(path.join(self.data_dir, 'transforms_all.json'), 'r') as fp:
			meta = json.load(fp)

		data_num = len(meta['image'])
		if self.num is None:
			self.data_list = list(range(data_num))
		elif isinstance(self.num, list) or isinstance(self.num, tuple):
			if self.split == 'train':
				self.data_list = self.num
			else:
				self.data_list = [x for x in list(range(data_num)) if x not in self.num]

		if len(self.data_list) < 10:
			print(f'[{self.split}] List: {self.data_list}')
		else:
			print(f'[{self.split}] List: {self.data_list[:5]} ... {self.data_list[-5:]}')

		cams = []
		for material in ['image', 'albedo', 'normal', 'depth']:
			images = []
			for i in self.data_list:
				frame = meta[material][i]
				fname = os.path.join(self.data_dir, frame['file_path'] + '.exr')
				with open(fname, 'rb') as imgin:
					image = read_exr(imgin)
					if self.factor > 0:
						[halfres_h, halfres_w] = [hw // self.factor for hw in image.shape[:2]]
						image = cv2.resize(image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
					else:
						raise ValueError('{} is negative, Please use positive number'.format(self.factor))

				if self.white_bkgd:
					image = image[..., :3] * image[..., -1:] + (1. - image[..., -1:])

				if material == 'image':
					mx = np.array(frame['transform_matrix'], dtype=np.float32)  # blender coordinate system
					rm = mx[:3, :3]

					if ('rot' in self.data_dir) or ('std' in self.data_dir):
						rm = bld_to_wd(rm)
						mx[:3, :3] = rm
					else:
						rm = np.eye(3)
						mx[:3, :3] = rm

					translate = mx[:3, -1].copy()
					mx[:3, -1] = translate @ bld_to_wd()
					cams.append(mx)

					image = np.nan_to_num(image, nan=0)
					image = np.clip(image[..., :3], 0, 1000)
					images.append(image)
				elif material == 'depth':
					if self.normalize_depth:
						image = np.clip(image[:, :, :1], self.near, self.far)
						image = (image - self.near) / (self.far - self.near)
					else:
						image = image[:, :, :1]
					images.append(image)
				elif material == 'normal':
					image = image * 2 - 1
					if 'pano' in self.data_dir:
						image = nor_to_nor(image[..., :3])
					images.append(image)
				else:
					images.append(image[..., :3])

			if material == 'image':
				self.images = images
				self.h, self.w = self.images[0].shape[:-1]
				self.camtoworlds = cams
				self.n_examples = len(self.images)
				del cams
			elif material == 'albedo':
				self.albedos = images
			elif material == 'normal':
				self.normals = images
			elif material == 'depth':
				self.depths = images

			del images
	
	def _train_init(self):
		"""Initialize training."""

		self._load_renderings()
		self._generate_rays()

		if self.split == 'train':
			self.images = self._flatten(self.images)
			self.depths = self._flatten(self.depths)
			self.normals = self._flatten(self.normals)
			self.albedos = self._flatten(self.albedos)
			self.rays = namedtuple_map(self._flatten, self.rays)
			if self.batch_type == 'all_images':
				self.num_samples = self.images.shape[0]
			else:
				self.num_samples = len(self.images)
		else:
			assert self.batch_type == 'single_image', 'The batch_type can only be single_image without flatten'

	def _generate_rays(self):
		if self.reform_cam:
			self.camtoworlds, self.origin, self.scale, self.rot = reform_c2w(np.array(self.camtoworlds), self.origin, self.scale, self.rot)

		"""Generating rays for all images."""
		theta, phi = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
			np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
			np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
			indexing='xy')

		# sphere direction
		theta = -(theta + 0.5) / self.w * 2 * np.pi
		phi = (phi + 0.5) / self.h * np.pi

		y = np.cos(phi)
		x = np.sin(phi) * np.sin(theta)
		z = np.sin(phi) * np.cos(theta)

		noise_range = np.sin(phi) * np.pi / self.w    # (h, w)
		noise_range = noise_range.reshape(self.h, self.w, 1)

		camera_dirs = np.stack([x, y, z], axis=-1)  # (h, w, 3)

		directions = [(camera_dirs @ c2w[:3, :3].T).copy() for c2w in self.camtoworlds]
		origins = [
			np.broadcast_to(c2w[:3, -1], v.shape).copy()
			for v, c2w in zip(directions, self.camtoworlds)
		]
	
		noise_var = [noise_range.copy() for _ in self.camtoworlds]
		viewdirs = [v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions]

		def broadcast_scalar_attribute(x):
			return [
				x * np.ones_like(origins[i][..., :1])
				for i in range(len(self.camtoworlds))
			]

		nears = broadcast_scalar_attribute(self.near).copy()
		fars = broadcast_scalar_attribute(self.far).copy()
		lossmults = broadcast_scalar_attribute(1).copy()

		# # original radii, (h, w, 3)
		# dx = [np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions]	# (h-1, w) * n
		# dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
		# radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

		# same radii for all pixels
		# breakpoint()
		dx = [np.tile(np.sqrt(np.sum((v[self.h // 2, :-1, :] - v[self.h // 2, 1:, :]) ** 2, -1))[None, :], (self.h, 1)) for v in directions]	# (h, w-1) * n
		dx = [np.concatenate([v, v[:, -2:-1]], 1) for v in dx]
		radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

		self.rays = Rays(
			origins=origins,
			directions=directions,
			viewdirs=viewdirs,
			radii=radii,
			lossmult=lossmults,
			near=nears,
			far=fars,
			noise_var=noise_var)
		
		self.radii = radii[0][0, 0, 0]
		del origins, directions, viewdirs, radii, lossmults, nears, fars, camera_dirs, noise_var

	def generate_lit_rays(self, num=80, near=0, far=10.0, type=torch.float16):
		points = []
		phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

		for i in range(num):
			y = 1 - (i / float(num - 1)) * 2  # y goes from 1 to -1
			radius = np.sqrt(1 - y * y)  # radius at y

			theta = phi * i  # golden angle increment

			x = np.cos(theta) * radius
			z = np.sin(theta) * radius

			points.append((x, y, z))

		directions = [np.array(points)]   # (n, 3)
		origins = [np.zeros_like(directions[0])]
		viewdirs = [v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions]

		def broadcast_scalar_attribute(x):
			return [x * np.ones_like(origins[0][..., :1])]

		lossmults = broadcast_scalar_attribute(4 * np.pi / num).copy()
		nears = broadcast_scalar_attribute(near).copy()
		fars = broadcast_scalar_attribute(far).copy()

		# same radii for all pixels
		radii = [np.tile(np.array(self.radii).reshape(1, 1), (num, 1))]

		env_rays = Rays(origins=origins,
		                directions=directions,
		                viewdirs=viewdirs,
		                radii=radii,
		                lossmult=lossmults,
		                near=nears,
		                far=fars,
		                noise_var=[np.zeros_like(nears[0]) for _ in nears])
		del origins, directions, viewdirs, radii, lossmults, nears, fars

		def _flatten(_x):
			_x = [_y.reshape([-1, _y.shape[-1]]) for _y in _x]
			return np.concatenate(_x, axis=0)

		env_rays = namedtuple_map(_flatten, env_rays)

		return Rays(*[torch.tensor(getattr(env_rays, key)).to(type) for key in Rays_keys])

	def put_on_device(self, rays, device):
		return namedtuple_map(lambda x: x.to(device), rays)

	def obtain_w2c(self, index):
		return np.array(self.camtoworlds[index])[:3, :3].T

	def __getitem__(self, index):
		if self.split == 'train' and self.batch_type == 'all_images':
			index = index % self.num_samples
		rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_keys])
		return rays, self.images[index], self.depths[index], self.normals[index], self.albedos[index]

	def __len__(self):
		if self.split == 'train'and self.batch_type == 'all_images':
			return 1000 * self.num_per_epoch
		else:
			return len(self.images)

	def get_cam_pos(self):
		return self.camtoworlds


def bld_to_wd(rm=None):
	b2w = R.from_euler('XYZ', [np.pi / 2, 0, 0]).as_matrix()
	if rm is None:
		return b2w  # T @ T = 1

	w2b = R.from_euler('XYZ', [-np.pi / 2, 0, 0]).as_matrix()
	align_center = R.from_euler('XYZ', [np.pi / 2, 0, 0]).as_matrix()

	return b2w.T @ rm @ w2b.T @ align_center


def nor_to_nor(x):
	return x @ R.from_euler('XYZ', [0, np.pi, 0]).as_matrix()


def scale_trans(c2w, origin=None, scale=None):
	trans = c2w[:, :3, -1].copy()	# xyz, (n, 3)

	if origin is None:
		origin = np.mean(trans, axis=0, keepdims=True)# (1, 3), mean of camera positions
	
	trans = trans - origin	# (n, 3)

	scale = 1
	c2w[:, :3, -1] = trans * scale

	return c2w, origin, scale


def rot_to_up(c2w, rot=None):
	if rot is None:
		rt = R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
		rms = c2w[:, :3, :3].copy()	# (3, 3)
		quats = [R.from_matrix(x).as_quat() for x in rms]
		mean_rm = R.from_quat(np.array(quats).mean(axis=0)).as_matrix()
		
		rot = (np.linalg.inv(mean_rm.T) @ rt.T).T

	for i in range(len(c2w)):
		c2w[i, :3, :4] = rot @ c2w[i, :3, :4].copy()

	return c2w, rot


def reform_c2w(c2w, origin=None, scale=None, rot=None):
	# c2w, rot = rot_to_up(c2w, rot)
	c2w, origin, scale = scale_trans(c2w, origin, scale)
	return c2w, origin, scale, rot