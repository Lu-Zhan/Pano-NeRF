import torch
import numpy as np


def sample_dir_by_pano(hw):
	h, w = hw
	theta, phi = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
		np.arange(w, dtype=np.float32),  # X-Axis (columns)
		np.arange(h, dtype=np.float32),  # Y-Axis (rows)
		indexing='xy')

	# sphere direction
	theta = -(theta + 0.5) / w * 2 * np.pi
	phi = (phi + 0.5) / h * np.pi

	y = np.cos(phi)
	x = np.sin(phi) * np.sin(theta)
	z = np.sin(phi) * np.cos(theta)

	return np.stack([x, y, z], axis=-1), theta, phi  # (h, w, 3)


def sample_dir_by_unifrom(num):
	points = []
	phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

	for i in range(num):
		y = 1 - (i / float(num - 1)) * 2  # y goes from 1 to -1
		radius = np.sqrt(1 - y * y)  # radius at y

		theta = phi * i  # golden angle increment

		x = np.cos(theta) * radius
		z = np.sin(theta) * radius

		points.append((x, y, z))

	return np.stack(points, axis=0)  # (num, 3)


def pos_to_spherical(pos):
	"""Convert the 3D position to spherical coordinates.
	Inputs:
		pos: (B, 3)
	Returns:
		theta: (B, 1)
		phi: (B, 1)
	"""
	if isinstance(pos, torch.Tensor):
		d = torch.norm(pos, dim=-1, keepdim=True)   # (B, 1)
		norm_pos = torch.nn.functional.normalize(pos, dim=-1)  # (B, 3)
		x, y, z = norm_pos[..., 0], norm_pos[..., 1], norm_pos[..., 2]
		t = (x ** 2 + z ** 2) ** 0.5
		phi = np.pi / 2 - torch.atan2(y, t)
		theta = torch.atan2(-x, -z) - np.pi
	elif isinstance(pos, np.ndarray):
		d = np.linalg.norm(pos, axis=-1, keepdims=True)  # (B, 1)
		norm_pos = pos / (d + 1e-8)  # (B, 3)
		x, y, z = norm_pos[..., 0], norm_pos[..., 1], norm_pos[..., 2]
		t = (x ** 2 + z ** 2) ** 0.5
		phi = np.pi / 2 - np.arctan2(y, t)
		theta = np.arctan2(-x, -z) - np.pi
	else:
		raise 'pos_to_spherical_coord only supports torch.Tensor and np.ndarray'

	return theta, phi, d


def spherical_to_pos(theta, phi, d=1):
	"""Convert the spherical coordinates to 3D position.
	Inputs:
		theta: (B, 1)
		phi: (B, 1)
		d: (B, 1)
	Returns:
		pos: (B, 3)
	"""
	if isinstance(theta, torch.Tensor):
		y = torch.cos(phi)
		x = torch.sin(phi) * torch.sin(theta)
		z = torch.sin(phi) * torch.cos(theta)
		pos = torch.cat([x, y, z], dim=-1) * d
	elif isinstance(theta, np.ndarray):
		y = np.cos(phi)
		x = np.sin(phi) * np.sin(theta)
		z = np.sin(phi) * np.cos(theta)
		pos = np.cat([x, y, z], axis=-1) * d
	else:
		raise 'spherical_coord_to_pos only supports torch.Tensor and np.ndarray'

	return pos


def spherical_to_pixel(theta, phi, hw=(128, 256)):
	"""Convert the spherical coordinates to pixel coordinates.
	Inputs:
		theta: (B, 1)
		phi: (B, 1)
		hw: (h, w)
	Returns:
		pixel: (B, 2)
	"""
	x = - theta / (2 * np.pi)
	y = phi / np.pi
	h, w = hw
	if isinstance(theta, torch.Tensor):
		pixel = torch.stack([w * x, h * y], dim=-1)
	elif isinstance(theta, np.ndarray):
		pixel = np.stack([w * x, h * y], axis=-1)
	else:
		raise 'spherical_coord_to_pixel only supports torch.Tensor and np.ndarray'

	return pixel


def interp_uni_to_pix(x, nums, scale=1):
	"""Interpolate the uniform sampled directions to the pixel coordinates.
	Inputs:
		x: (n, 3)
		nums: [k], e.g., [1, 3, 5, 3, 1]
		scale: int
	Output:
		xs: (h, w, 3)
	"""
	xs = []
	w = int(max(nums) / scale)
	for num in nums:
		num = int(num)
		index = num * (np.arange(w) + 0.5) / w

		line = []
		for j in range(3):
			line += [np.interp(index, np.arange(num), x[:num, j])]
		line = np.stack(line, axis=-1)

		xs.append(line)
		x = x[num:]
	xs = np.concatenate(xs, axis=0).reshape(-1, w, 3)

	return xs


def inv_uni_to_pix(x, map):
	"""Inverse mapping to px from uni
	Inputs:
		x: (n, 3)
		map: (h, w)
	Output:
		img: (h, w, 3)
	"""
	h, w = map.shape
	idx = map.reshape(-1)   # (h * w)
	img = x[idx, :]    # (h * w, 3)
	return img.reshape(h, w, 3)


if __name__ == '__main__':
	dir, tt, pp = sample_dir_by_pano((16, 32))
	t, p, d = pos_to_spherical(dir)
	rec_dir = spherical_to_pos(t, p, d)
	px = spherical_to_pixel(t, p, 16)
	print('test')