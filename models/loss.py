import torch

def thin_surface(n, d, weight):
	NoD = torch.bmm(n.view(-1, 1, 3), d.view(-1, 3, 1)).view(-1, 1)
	NoD = torch.relu(NoD)

	return weight.view(-1, 1) * NoD




