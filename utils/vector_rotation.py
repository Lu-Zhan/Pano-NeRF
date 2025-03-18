import numpy as np
import torch
from numpy.linalg import norm
from torch.nn.functional import normalize
from torch.func import vmap


def rot_to_target_np(target_vec, origin_vec=np.array([0, 1, 0])):
	if np.array_equal(origin_vec, -target_vec):
		return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
	theta = np.arccos(np.dot(origin_vec, target_vec) / (norm(origin_vec) * norm(target_vec)))
	n_vec = np.cross(origin_vec, target_vec)
	n_vec = n_vec / norm(n_vec)

	n_vec_invert = np.array([[0, -n_vec[2], n_vec[1]],
	                         [n_vec[2], 0, -n_vec[0]],
	                         [-n_vec[1], n_vec[0], 0]])
	I = np.eye(3)

	Rm = I + np.sin(theta) * n_vec_invert + n_vec_invert @ (n_vec_invert) * (1 - np.cos(theta))

	return Rm


def rot_to_target(target_vec, origin_vec=torch.Tensor([0, 1, 0])):
	device = target_vec.device
	dtype = target_vec.dtype
	origin_vec = origin_vec.to(device).to(dtype)
	if torch.equal(origin_vec, -target_vec):
		return torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).to(dtype)

	theta = torch.acos(torch.dot(origin_vec, target_vec))
	n_vec = torch.cross(origin_vec, target_vec)
	n_vec = normalize(n_vec.view(-1, 1)).view(-1)

	# n_vec_invert = torch.Tensor([[0, -n_vec[2], n_vec[1]],
	#                          [n_vec[2], 0, -n_vec[0]],
	#                          [-n_vec[1], n_vec[0], 0]], device=device).to(dtype)

	a = torch.Tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
	                  [0, 0, 1, 0, 0, 0, -1, 0, 0],
	                  [0, -1, 0, 1, 0, 0, 0, 0, 0]], device=device).to(dtype)
	n_vec_invert = (n_vec @ a).view(3, 3)

	I = torch.eye(3, device=device).to(dtype)

	return I + torch.sin(theta) * n_vec_invert + n_vec_invert @ (n_vec_invert) * (1 - torch.cos(theta))


class RotToTarget():
	def __init__(self):
		self.a = torch.Tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
	                  [0, 0, 1, 0, 0, 0, -1, 0, 0],
	                  [0, -1, 0, 1, 0, 0, 0, 0, 0]]).view(1, 3, 9)
		self.ovec = torch.Tensor([0, 1, 0]).view(1, 3)
		self.rm_inverse = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
		self.start_flag = False

	def rot2t(self, tvec):
		"""Compute the batched rotation matrix to rotate the origin vector to the target vector.
		Input:
			tvec: (B, 3)
		Output:
			rot: (B, 3, 3)
		"""
		if self.start_flag == False:
			self.device = tvec.device
			self.dtype = tvec.dtype
			self.ovec = self.ovec.to(self.device).to(self.dtype)
			self.a = self.a.to(self.device).to(self.dtype)
			self.rm_inverse = self.rm_inverse.to(self.device).to(self.dtype)
			self.start_flag = True

		theta = torch.acos(torch.matmul(self.ovec.view(1, 1, 3), tvec.view(-1, 3, 1)))  # (1, 1, 3) @ (B, 3, 1) = (B, 1, 1)
		indice = torch.where(theta == np.pi)[0]
		n_vec = torch.cross(self.ovec, tvec) # (1, 3) x (B, 3) = (B, 3)
		n_vec = normalize(n_vec)  # (B, 3)

		# shew-symmetric matrix of n_vec, (B, 1, 3) @ (B, 3, 9) = (B, 1, 9) -> (B, 3, 3)
		n_vec_invert = torch.matmul(n_vec.view(-1, 1, 3), self.a).view(-1, 3, 3)
		I = torch.eye(3, device=self.device).to(self.dtype).view(1, 3, 3)

		mm_nvi = torch.bmm(n_vec_invert, n_vec_invert)  # (B, 3, 3) @ (B, 3, 3) = (B, 3, 3)

		# (1, 3, 3) + (B, 1, 1) * (B, 3, 3) + (B, 3, 3) * (B, 1, 1) = (B, 3, 3)
		rm = I + torch.sin(theta) * n_vec_invert + mm_nvi * (1 - torch.cos(theta))
		rm[indice] = self.rm_inverse

		return rm


if __name__ == '__main__':
	# sampling
	theta, phi = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
		np.arange(10, dtype=np.float32),  # X-Axis (columns)
		np.arange(10, dtype=np.float32),  # Y-Axis (rows)
		indexing='xy')

	# sphere direction
	theta = -(theta + 0.5) / 10 * 2 * np.pi
	phi = (phi + 0.5) / 10 * np.pi
	y = np.cos(phi)
	x = np.sin(phi) * np.sin(theta)
	z = np.sin(phi) * np.cos(theta)
	camera_dirs = np.stack([x, y, z], axis=-1)  # (h, w, 3)
	ps = camera_dirs.reshape(-1, 3)[:50]  # (h*w/2, 3)
	ps = torch.from_numpy(ps)

	# rotation
	target_vec = np.array([[0, -1., 0]])
	target_vec = torch.from_numpy(target_vec).float()
	Rot = RotToTarget()
	Rm = Rot.rot2t(target_vec.view(-1, 3)).view(3, 3)
	# Rm = rot_to_target(target_vec)

	# visualization
	from matplotlib import pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], c='r', marker='o')

	r_ps = ps @ Rm.t()
	ax.scatter(r_ps[:, 0], r_ps[:, 1], r_ps[:, 2], c='b', marker='o')

	plt.show()