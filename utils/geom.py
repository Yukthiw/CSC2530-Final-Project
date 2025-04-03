# reference: https://github.com/aharley/simple_bev/blob/main/utils/geom.py
import numpy as np
import torch

'''
This is all taken from BEVCAR Repo, necessary for voxelization.
'''

def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1, 4, 4).repeat([B, 1, 1])
    return rt

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:, :, 0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    # xyz2 = xyz2 / xyz2[:,:,3:4]
    xyz2 = xyz2[:, :, :3]
    return xyz2