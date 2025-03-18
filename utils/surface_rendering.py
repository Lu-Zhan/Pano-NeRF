import torch
import numpy as np
import torch.nn.functional as f
from einops import repeat, rearrange

def microfeast_brdf(albedo, normal, roughness, l, v):
	"""BRDF.
	Reference: "Real shading in unreal engine 4, 2013"
    Args:
        albedo: torch.Tensor, base color, [batch_size, 3]
        normal: torch.Tensor, surface normal, [batch_size, 3]
		roughness: torch.Tensor, specular roughness, [batch_size, 1]
		l: torch.Tensor, light direction, [batch_size, lit_dir, 3]
		v: torch.Tensor, view direction, [batch_size, 3]
    Returns:
        diffuse_brdf: torch.Tensor, ratio, [batch_size, lit_dir, 3]
        specular_brdf: torch.Tensor, ratio, [batch_size, lit_dir, 3]
        NoL: torch.Tensor, ratio, [batch_size, lit_dir, 1]
    """

	num_lit = l.shape[1]

	# diffuse term
	diffuse_brdf = albedo / np.pi
	diffuse_brdf = repeat(diffuse_brdf, 'b c -> b d c', d=num_lit)

	# helf vector
	v = repeat(v, 'b c -> b d c', d=num_lit).reshape(-1, 3)
	n = repeat(normal, 'b c -> b d c', d=num_lit).reshape(-1, 3)
	r = repeat(roughness, 'b c -> b d c', d=num_lit).reshape(-1, 1)
	l = l.view(-1, 3)
	h = l + v
	h = f.normalize(h, dim=-1)
	NoH = torch.bmm(n.view(-1, 1, 3), h.view(-1, 3, 1)).squeeze(-1)
	VoH = torch.bmm(v.view(-1, 1, 3), h.view(-1, 3, 1)).squeeze(-1)
	NoL = torch.bmm(n.view(-1, 1, 3), l.view(-1, 3, 1)).squeeze(-1)
	NoV = torch.bmm(n.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1)

	NoH = torch.relu(NoH)
	VoH = torch.relu(VoH)
	NoL = torch.relu(NoL)
	NoV = torch.relu(NoV)

	F0 = 0.04 # refer to UE4

	# specular term
	# function D
	alpha = r ** 2
	# k = (r + 1) ** 2 / 8  # disney brdf for analytic light source
	k = r ** 2 / 2  # for IBL
	D = alpha ** 2 / (np.pi * ((NoH ** 2) * (alpha ** 2 - 1) + 1) ** 2)
	F = F0 + (1 - F0) * 2 ** (-(5.55473 * VoH + 6.98316) * VoH)
	G1 = NoL / ((1 - k) * NoL + k)
	G2 = NoV / ((1 - k) * NoV + k)
	G = G1 * G2

	specular_brdf = D * F * G / (4 * NoL * NoV)
	specular_brdf = specular_brdf.nan_to_num(nan=0, posinf=0)
	specular_brdf = specular_brdf.view(-1, num_lit, 1)

	return diffuse_brdf, specular_brdf, NoL.view(-1, num_lit, 1)


def blinn_phong_brdf(albedo, normal, roughness, l, v):
	"""blinn-phong BRDF.
	Reference: "Phong Reflectance Model, 1975"
    Args:
        albedo: torch.Tensor, base color, [batch_size, 3]
        normal: torch.Tensor, surface normal, [batch_size, 3]
		roughness: torch.Tensor, specular roughness, [batch_size, 1]
		l: torch.Tensor, light direction, [batch_size, lit_dir, 3]
		v: torch.Tensor, view direction, [batch_size, 3]
    Returns:
        diffuse_brdf: torch.Tensor, ratio, [batch_size, lit_dir, 3]
        specular_brdf: torch.Tensor, ratio, [batch_size, lit_dir, 3]
        NoL: torch.Tensor, ratio, [batch_size, lit_dir, 1]
    """

	num_lit = l.shape[1]

	# diffuse term
	diffuse_brdf = albedo / np.pi
	diffuse_brdf = repeat(diffuse_brdf, 'b c -> b d c', d=num_lit)

	# specular term
	# helf vector
	v = repeat(v, 'b c -> b d c', d=num_lit).reshape(-1, 3)
	n = repeat(normal, 'b c -> b d c', d=num_lit).reshape(-1, 3)
	r = repeat(roughness, 'b c -> b d c', d=num_lit).reshape(-1, 1)
	l = l.view(-1, 3)
	h = l + v
	h = f.normalize(h, dim=-1)
	NoH = torch.bmm(n.view(-1, 1, 3), h.view(-1, 3, 1)).squeeze(-1) # [batch_size * lit_dir, 1]
	NoL = torch.bmm(n.view(-1, 1, 3), l.view(-1, 3, 1)).squeeze(-1) # [batch_size * lit_dir, 1]
	NoH = torch.relu(NoH)

	specular_brdf = torch.pow(NoH, r)
	specular_brdf = specular_brdf.nan_to_num(nan=0, posinf=0)
	specular_brdf = specular_brdf.view(-1, num_lit, 1)

	return diffuse_brdf, specular_brdf, NoL.view(-1, num_lit, 1)


def lambertian_brdf(albedo, normal, l, cos_th=0):
	"""Lambertian BRDF.
	Args:
		albedo: torch.Tensor, base color, [batch_size, 3]
		normal: torch.Tensor, surface normal, [batch_size, 3]
		l: torch.Tensor, light direction, [batch_size, lit_dir, 3]
	Returns:
		diffuse_brdf: torch.Tensor, ratio, [batch_size, lit_dir, 3]
		NoL: torch.Tensor, ratio, [batch_size, lit_dir, 1]
	"""
	num_lit = l.shape[1]

	# diffuse term
	diffuse_brdf = albedo / np.pi   # [batch_size, 3]
	# diffuse_brdf = repeat(diffuse_brdf, 'b c -> b d c', d=num_lit)

	# NoL
	n = repeat(normal, 'b c -> b d c', d=num_lit).reshape(-1, 3)

	NoL = torch.bmm(n.view(-1, 1, 3), l.view(-1, 3, 1)).squeeze(-1)
	NoL = torch.relu(NoL - cos_th) + cos_th

	return diffuse_brdf, NoL.view(-1, num_lit, 1)


def surface_rendering(env, albedo, normal, roughness, l, v, solid_angle, output_sd=False):
	"""Surface Rendering Function.
    Args:
    	env: torch.Tensor, env lighting, [batch_size, lit_dir, 3]
        albedo: torch.Tensor, base color, [batch_size, 3]
        normal: torch.Tensor, surface normal, [batch_size, 3]
		roughness: torch.Tensor, specular roughness, [batch_size, 1]
		l: torch.Tensor, light direction, [batch_size, lit_dir, 3]
		v: torch.Tensor, view direction, [batch_size, 3]
		solid_angle: torch.Tensor, solid angle, [1, lit_dir, 1]
    Returns:
    	rgb: torch.Tensor, color, [batch_size, 3]
    	diffuse: torch.Tensor, diffuse term, [batch_size, 3]
    	specular: torch.Tensor, specular term, [batch_size, 3]
    """

	shading = None

	if roughness is not None:
		diffuse_brdf, specular_brdf, NoL = microfeast_brdf(albedo, normal, roughness, l, v)
		diffuse = torch.sum(diffuse_brdf * env * NoL * solid_angle, dim=1)
		specular = torch.sum(specular_brdf * env * solid_angle, dim=1)
	else:
		diffuse_brdf, NoL = lambertian_brdf(albedo, normal, l, cos_th=0)
		shading = torch.sum(env * NoL * solid_angle, dim=1)   # [batch_size, 3], shading for each pixel
		diffuse = diffuse_brdf * shading
		specular = torch.zeros_like(diffuse)

	rgb = diffuse + specular
	if not output_sd:
		return rgb, diffuse, specular
	else:
		if shading is not None:
			return rgb, diffuse, specular, shading
		else:
			print('[Warning] Shading is None')
			return rgb, diffuse, specular, shading


def surface_rendering_wlit(env, env_weight, albedo, normal, roughness, l, v, solid_angle, output_sd=False):
	"""Surface Rendering Function.
    Args:
    	env: torch.Tensor, env lighting, [batch_size, k, lit_dir, 3]
        env_weight: torch.Tensor, env weight, [batch_size, k]
        albedo: torch.Tensor, base color, [batch_size, 3]
        normal: torch.Tensor, surface normal, [batch_size, 3]
		roughness: torch.Tensor, specular roughness, [batch_size, 1]
		l: torch.Tensor, light direction, [batch_size, lit_dir, 3]
		v: torch.Tensor, view direction, [batch_size, 3]
		solid_angle: torch.Tensor, solid angle, [1, lit_dir, 1]
    Returns:
    	rgb: torch.Tensor, color, [batch_size, 3]
    	diffuse: torch.Tensor, diffuse term, [batch_size, 3]
    	specular: torch.Tensor, specular term, [batch_size, 3]
    """

	assert roughness is None, 'Not implemented yet'

	diffuse_brdf, NoL = lambertian_brdf(albedo, normal, l, cos_th=0)
	NoL = rearrange(NoL, 'b d c -> b 1 d c')
	solid_angle = rearrange(solid_angle, 'd c -> 1 1 d c')
	shading = torch.sum(env * NoL * solid_angle, dim=2)   # [batch_size, k, 3], shading for each pixel per env
	shading = torch.sum(shading * env_weight.unsqueeze(-1), dim=1)   # [batch_size, 3], shading for each pixel
	diffuse = diffuse_brdf * shading
	specular = torch.zeros_like(diffuse)
	rgb = diffuse

	if not output_sd:
		return rgb, diffuse, specular
	else:
		if shading is not None:
			return rgb, diffuse, specular, shading
		else:
			print('[Warning] Shading is None')
			return rgb, diffuse, specular, shading


def surface_rendering_hemi(env, env_weight, albedo, NoL, solid_angle, output_sd=False):
	"""Surface Rendering Function, using hemispherical lighting and fixed NoL.
    Args:
    	env: torch.Tensor, env lighting, [batch_size, k, lit_dir, 3]
        env_weight: torch.Tensor, env weight, [batch_size, k]
        albedo: torch.Tensor, base color, [batch_size, 3]
		NoL: torch.Tensor, [lit_dir, 1]
		solid_angle: torch.Tensor, solid angle, [1, lit_dir, 1]
		output_sd: bool, whether to output shading
    Returns:
    	rgb: torch.Tensor, color, [batch_size, 3]
    	diffuse: torch.Tensor, diffuse term, [batch_size, 3]
    	specular: None
    	shading: torch.Tensor, shading term, [batch_size, 3]
    """

	diffuse_brdf = albedo / np.pi
	NoL = rearrange(NoL, 'd c -> 1 1 d c')
	solid_angle = rearrange(solid_angle, 'd c -> 1 1 d c')
	shading = torch.sum(env * NoL * solid_angle, dim=2)   # [batch_size, k, 3], shading for each pixel per env
	shading = torch.sum(shading * env_weight.unsqueeze(-1), dim=1)   # [batch_size, 3], shading for each pixel
	diffuse = diffuse_brdf * shading
	# specular = torch.zeros_like(diffuse)
	rgb = diffuse

	if not output_sd:
		return rgb, diffuse, None
	else:
		return rgb, diffuse, None, shading


def surface_rendering_point_lit(point_lit, albedo, normal, position, output_sd=False):
	"""Surface Rendering Function, using 3d point lighting.
	Args:
		point_lit: torch.Tensor, spherical gaussian lighting, [num_lit, 3+3+1+1], color, dir, dist, stedradian
		albedo: torch.Tensor, base color, [batch_size, 3]
		normal: torch.Tensor, surface normal, [batch_size, 3]
		position: torch.Tensor, surface position, [batch_size, 3]
		output_sd: bool, whether to output shading
	Returns:
		rgb: torch.Tensor, color, [batch_size, 3]
		diffuse: torch.Tensor, diffuse term, [batch_size, 3]
		specular: torch.Tensor, specular term, [batch_size, 3]
	"""

	point_lit = wrap_sg_lit(point_lit, position)    # (b, n, 3+3+1+1)
	c = point_lit[:, :, :3]    # (b, n, 3)
	l = point_lit[:, :, 3:6]    # (b, n, 3)
	s = point_lit[:, :, 7:8]    # (b, n, 1)

	diffuse_brdf, NoL = lambertian_brdf(albedo, normal, l, cos_th=0)    # (b, n, 3), (b, n, 1)
	shading = torch.sum(c * NoL * s, dim=1)    # (b, 3), shading for each pixel
	diffuse = diffuse_brdf * shading
	specular = torch.zeros_like(diffuse)

	if not output_sd:
		return diffuse, diffuse, specular
	else:
		return diffuse, diffuse, specular, shading


def wrap_sg_lit(sg_lit, position):
	"""Wrap spherical gaussian lighting.
	Args:
		sg_lit: torch.Tensor, spherical gaussian lighting, [num_lit, 3+3+1+1], color, dir, dist, stedradian
		position: torch.Tensor, surface position, [batch_size, 3]
	Returns:
		sg_lit: torch.Tensor, [batch_size, num_lit, 3], batched lits with new dir and dist
	"""

	lit_col = sg_lit[:, :3] # (n, 3)
	lit_dir = sg_lit[:, 3:6] # (n, 3)
	lit_dist = sg_lit[:, 6:7] # (n, 1)
	lit_steradian = sg_lit[:, 7:8] # (n, 1)

	lit_pos = lit_dir * lit_dist
	new_vec = lit_pos.unsqueeze(0) - position.unsqueeze(1)    # (b, n, 3)
	new_dist = torch.norm(new_vec, dim=-1, keepdim=True)    # (b, n, 1)
	new_dir = f.normalize(new_vec, dim=-1)    # (b, n, 3)

	new_steradian = lit_steradian.unsqueeze(0) * lit_dist.unsqueeze(0) ** 2 / (new_dist ** 2 + 1e-8)   # (b, n, 1)

	b = position.shape[0]
	lit_col = lit_col.unsqueeze(0).expand(b, -1, -1)    # (b, n, 3)

	return torch.cat([lit_col, new_dir, new_dist, new_steradian], dim=-1)


def solid_angle_refinement(h=8, w=16, hemisp=False, type='torch'):
	"""Solid Angle Refinement.
	Args:
		h: int, height
		w: int, width
		hemisp: bool, whether to use hemispherical or spherical
	Returns:
		solid_angle: torch.Tensor, solid angle, [lit_dir, 1] from [h, w]
	"""
	phi_range = np.pi / 2 if hemisp else np.pi
	d_phi = phi_range / h
	d_theta = 2 * np.pi / w

	y = (np.arange(h) + 0.5) / h
	x = (np.arange(w) + 0.5) / w

	_, yy = np.meshgrid(x, y)
	sin_phi = np.sin(yy * phi_range)

	solid_angle = sin_phi * d_theta * d_phi
	solid_angle = solid_angle.reshape(1, -1, 1)

	return torch.Tensor(solid_angle) if type == 'torch' else solid_angle.reshape(h, w, 1)


def hdr_to_ldr(color, gamma=2.2, dtype='float32', clamp=True):
	# aces tone mapping
	A = 2.51
	B = 0.03
	C = 2.43
	D = 0.59
	E = 0.14

	# color: [B, 3]
	color = (color * (A * color + B)) / (color * (C * color + D) + E)
	if type(color) is torch.Tensor:
		if clamp:
			color = torch.clamp(color, 0, 1)
		if dtype == 'uint8':
			color = (color * 255.).to(torch.uint8) / 255.
			color = color.to(torch.float32)
	elif type(color) is np.ndarray:
		if clamp:
			color = np.clip(color, 0, 1)
		if dtype == 'uint8':
			color = (color * 255).astype(np.uint8) / 255.
			color = color.astype(np.float32)
	else:
		raise TypeError('color must be torch.Tensor or np.ndarray')

	return color ** (1 / gamma)


def compute_illumination(x):
	op = torch.Tensor([0.2126, 0.7152, 0.0722]).reshape(3, 1).to(x.device).to(x.dtype)

	if x.shape[-1] == 3:
		return torch.matmul(x, op)  # [..., 3] @ [3, 1] -> [..., 1]
	else:
		x = rearrange(x, 'b c h w -> b h w c')
		return torch.matmul(x, op)  # [..., 3] @ [3, 1] -> [..., 1]


def visual_tonemapping():
	import matplotlib.pyplot as plt

	x = np.arange(0, 20, 0.01)  # (20,)
	y = hdr_to_ldr(x, gamma=1, clamp=False)  # (20,)
	y_ga = hdr_to_ldr(x, gamma=2.2, clamp=False)  # (20,)

	plt.figure()
	plt.subplot(2, 1, 1)
	plt.plot(x, y)
	plt.title('gamma=1')
	plt.subplot(2, 1, 2)
	plt.plot(x, y_ga)
	plt.title('gamma=2.2')
	plt.show()


# Unit test
def render_unit_test():
	import matplotlib.pyplot as plt
	from PIL import Image
	from os.path import join
	from torchvision import transforms as T

	tf = T.Compose([T.Resize(100), T.ToTensor()])

	root_dir = '../../Database/phong_sample'
	img = Image.open(join(root_dir, 'image_wo_shd_orange.png')).convert('RGB')
	nor = Image.open(join(root_dir, 'normal.png')).convert('RGB')
	alb = Image.open(join(root_dir, 'albedo.png')).convert('RGB')
	pos = Image.open(join(root_dir, 'position.png')).convert('RGB')

	img = tf(img).permute(1, 2, 0).view(-1, 3)
	nor = tf(nor).permute(1, 2, 0).view(-1, 3) * 2 - 1
	alb = tf(alb).permute(1, 2, 0).view(-1, 3)
	pos = tf(pos).permute(1, 2, 0).view(-1, 3) * 2 - 1

	# transfer coordinate system, x-> -x, z -> -z
	# pos = pos * torch.tensor([-1, 1, -1]).view(1, 3)
	# nor = nor * torch.tensor([-1, 1, -1]).view(1, 3)

	# view direction
	v = torch.tensor([0, 0, 1.7]).view(1, 3).float() - pos

	# lighting
	env = torch.tensor([1, 1, 1]).view(1, 1, 3).float()
	lit_dir = torch.tensor([[1, 1, 1]]).view(1, 1, 3).float()
	env = env.repeat(nor.shape[0], 1, 1)
	lit_dir = lit_dir.repeat(nor.shape[0], 1, 1)

	# render
	diffuse, NoL = lambertian_brdf(alb, nor, lit_dir)
	diffuse = torch.sum(diffuse * env * NoL, dim=1)

	plt.figure()

	def plot(_img):
		plt.imshow(_img.numpy().reshape(100, 100, 3))
		plt.show()

	plot(diffuse)


def simple_render_test():
	import matplotlib.pyplot as plt
	from PIL import Image
	from os.path import join
	from torchvision import transforms as T

	h, w = 256, 256
	xx, yy = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
		np.arange(w, dtype=np.float32),  # X-Axis (columns)
		np.arange(h, dtype=np.float32),  # Y-Axis (rows)
		indexing='xy')

	xx = (xx - w / 2 + 0.5) / w * 2
	yy = (h / 2 - yy - 0.5) / h * 2

	zz = 1 - xx ** 2 - yy ** 2
	zz = np.maximum(zz, 0) ** 0.5
	mask = zz != 0

	mask = torch.from_numpy(mask).float().view(-1, 1)
	pos = np.stack([xx, yy, zz], axis=2)
	pos = torch.from_numpy(pos).float().view(-1, 3)
	nor = pos.clone()

	# view direction
	v = torch.tensor([0, 0, 1.7]).view(1, 3).float() - pos

	# lighting
	env = torch.tensor([1, 1, 1]).view(1, 1, 3).float()
	lit_dir = torch.tensor([[1, 1, 1]]).view(1, 1, 3).float()
	env = env.repeat(nor.shape[0], 1, 1)
	lit_dir = lit_dir.repeat(nor.shape[0], 1, 1)

	# render
	albedo = torch.ones_like(pos)
	roughness = torch.ones_like(pos[..., :1]) * 10
	diffuse_brdf, specular_brdf, NoL = microfeast_brdf(albedo, nor, roughness, lit_dir, v)
	diffuse = torch.sum(diffuse_brdf * env * NoL, dim=1)
	specular = torch.sum(specular_brdf * env * NoL, dim=1)

	plt.figure()
	def plot(_img):
		plt.imshow(_img.numpy().reshape(256, 256, 3))
		plt.show()

	# plot(diffuse * mask)
	plot(specular * mask)
	plot((diffuse + specular) * mask)

	print((specular * mask).max())
	print((diffuse * mask).max())
	print(((diffuse + specular) * mask).max())

	print('done')


def sglit_render_unit_test():
	import matplotlib.pyplot as plt
	from PIL import Image
	from os.path import join
	from torchvision import transforms as T

	h, w = 256, 256
	xx, yy = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
		np.arange(w, dtype=np.float32),  # X-Axis (columns)
		np.arange(h, dtype=np.float32),  # Y-Axis (rows)
		indexing='xy')

	xx = (xx - w / 2 + 0.5) / w * 2
	yy = (h / 2 - yy - 0.5) / h * 2

	zz = 1 - xx ** 2 - yy ** 2
	zz = np.maximum(zz, 0) ** 0.5
	mask = zz != 0

	mask = torch.from_numpy(mask).float().view(-1, 1)
	pos = np.stack([xx, yy, zz], axis=2)
	pos = torch.from_numpy(pos).float().view(-1, 3) # (256*256, 3)
	nor = pos.clone()
	alb = torch.ones_like(pos)

	# view direction
	v = torch.tensor([0, 0, 1.7]).view(1, 3).float() - pos

	# lighting
	c = torch.tensor([10, 10, 10]).view(1, 3).float()
	l = torch.tensor([[1, 1, 1]]).view(1, 3).float()
	d = torch.tensor([8]).view(1, 1).float()
	s = torch.tensor([0.5]).view(1, 1).float()

	sg_lit = torch.cat([c, l, d, s], dim=1) # (1, 8)

	# render
	_, diffuse, specular = surface_rendering_point_lit(sg_lit, alb, nor, pos, False)

	plt.figure()
	def plot(_img):
		plt.imshow(_img.numpy().reshape(256, 256, 3))
		plt.show()

	# plot(diffuse * mask)
	# plot(specular * mask)
	plot((diffuse + specular) * mask)

	# print((specular * mask).max())
	print((diffuse * mask).max())
	# print(((diffuse + specular) * mask).max())

	print('done')


if __name__=='__main__':
	# visual_tonemapping()
	sglit_render_unit_test()