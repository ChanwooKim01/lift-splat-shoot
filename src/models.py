"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import numpy as np

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True).to(torch.device("cuda:0"))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ).to(torch.device("cuda:0"))

    def forward(self, x1, x2):
        x1 = x1.to(torch.device("cuda:0"))
        x2 = x2.to(torch.device("cuda:0"))
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1).to(torch.device("cuda:0"))
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0").to(torch.device("cuda:0"))

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0).to(torch.device("cuda:0"))

    def get_depth_dist(self, x, eps=1e-20):
        x = x.to(torch.device("cuda:0"))
        return x.softmax(dim=1).to(torch.device("cuda:0"))

    def get_depth_feat(self, x):
        x = x.to(torch.device("cuda:0"))
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        new_x = new_x.to(torch.device("cuda:0"))

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x))).to(torch.device("cuda:0"))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate).to(torch.device("cuda:0"))
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False).to(torch.device("cuda:0"))
        self.bn1 = trunk.bn1.to(torch.device("cuda:0"))
        self.relu = trunk.relu.to(torch.device("cuda:0"))

        self.layer1 = trunk.layer1.to(torch.device("cuda:0"))
        self.layer2 = trunk.layer2.to(torch.device("cuda:0"))
        self.layer3 = trunk.layer3.to(torch.device("cuda:0"))

        self.up1 = Up(64+256, 256, scale_factor=4).to(torch.device("cuda:0"))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True).to(torch.device("cuda:0")),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False).to(torch.device("cuda:0")),
            nn.BatchNorm2d(128).to(torch.device("cuda:0")),
            nn.ReLU(inplace=True).to(torch.device("cuda:0")),
            nn.Conv2d(128, outC, kernel_size=1, padding=0).to(torch.device("cuda:0")),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x.to(torch.device("cuda:0"))


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        # self.dx = nn.Parameter(dx, requires_grad=False).to(torch.device("cuda:0"))
        # self.bx = nn.Parameter(bx, requires_grad=False).to(torch.device("cuda:0"))
        # self.nx = nn.Parameter(nx, requires_grad=False).to(torch.device("cuda:0"))
        self.register_buffer('dx', dx)
        self.register_buffer('bx', bx)
        self.register_buffer('nx', nx)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum().to(torch.device("cuda:0"))
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        # self.use_quickcumsum = True
        self.use_quickcumsum = False
        
    # for onnx export
    # def inv(self, S):
    #    Sinv = np.linalg.inv(S.detach().cpu().numpy())
    #    return torch.from_numpy(Sinv).to(S.device)
    def safe_inverse_3x3(self, M, eps=1e-8):
        a,b,c = M[...,0,0], M[...,0,1], M[...,0,2]
        d,e,f = M[...,1,0], M[...,1,1], M[...,1,2]
        g,h,i = M[...,2,0], M[...,2,1], M[...,2,2]
        A = e*i - f*h
        B = -(d*i - f*g)
        C = d*h - e*g
        D = -(b*i - c*h)
        E = a*i - c*g
        F = -(a*h - b*g)
        G = b*f - c*e
        H = -(a*f - c*d)
        I = a*e - b*d
        det = a*A + b*B + c*C
        det = det.clamp(min=eps)
        inv = torch.stack([
            torch.stack([A, D, G], -1),
            torch.stack([B, E, H], -1),
            torch.stack([C, F, I], -1)
        ], -2)
        return (inv / det.unsqueeze(-1).unsqueeze(-1)).to(torch.device("cuda:0"))
        
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW).to(torch.device("cuda:0"))
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW).to(torch.device("cuda:0"))
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW).to(torch.device("cuda:0"))

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1).to(torch.device("cuda:0"))
        return nn.Parameter(frustum, requires_grad=False).to(torch.device("cuda:0"))

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = points.to(torch.device("cuda:0"))
        # points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # points = self.inv(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        post_rots_inv = self.safe_inverse_3x3(post_rots).to(torch.device("cuda:0"))
        points = post_rots_inv.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).to(torch.device("cuda:0"))
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5).to(torch.device("cuda:0"))
        # combine = rots.matmul(torch.inverse(intrins))
        
        # combine = rots.matmul(self.inv(intrins))
        intrins_inv = self.safe_inverse_3x3(intrins).to(torch.device("cuda:0"))
        combine = rots.matmul(intrins_inv).to(torch.device("cuda:0"))
        
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1).to(torch.device("cuda:0"))
        points += trans.view(B, N, 1, 1, 1, 3)

        return points.to(torch.device("cuda:0"))

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x).to(torch.device("cuda:0"))
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample).to(torch.device("cuda:0"))
        x = x.permute(0, 1, 3, 4, 5, 2).to(torch.device("cuda:0"))

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C).to(torch.device("cuda:0"))

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long().to(torch.device("cuda:0"))
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)]).to(torch.device("cuda:0"))
        geom_feats = torch.cat((geom_feats, batch_ix), 1).to(torch.device("cuda:0"))

        # filter out points that are outside box
        # kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
        #     & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
        #     & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        nx_val = self.nx.to(x.device)
        zeros = torch.zeros_like(geom_feats[:, 0]).to(torch.device("cuda:0"))
        kept_x = (geom_feats[:, 0] >= zeros) & (geom_feats[:, 0] < nx_val[0])
        kept_y = (geom_feats[:, 1] >= zeros) & (geom_feats[:, 1] < nx_val[1])
        kept_z = (geom_feats[:, 2] >= zeros) & (geom_feats[:, 2] < nx_val[2])

        kept = kept_x & kept_y & kept_z
        x = x[kept].to(torch.device("cuda:0"))
        geom_feats = geom_feats[kept].to(torch.device("cuda:0"))
        
        # Use a fully ONNX-compatible method (One-Hot + MatMul) for aggregation.
        nx_x, nx_y, nx_z = nx_val[0], nx_val[1], nx_val[2]
        K = int(nx_x * nx_y * nx_z)

        ranks = geom_feats[:, 3] * K + \
                geom_feats[:, 2] * (nx_x * nx_y) + \
                geom_feats[:, 0] * nx_y + \
                geom_feats[:, 1]
        ranks = ranks.long().to(torch.device("cuda:0"))

        one_hot = torch.nn.functional.one_hot(ranks, num_classes=B * K).float().to(torch.device("cuda:0"))
        
        final_very_flat = (one_hot.transpose(0, 1) @ x).to(torch.device("cuda:0"))

        final_flat = final_very_flat.view(B, K, C).permute(0, 2, 1).to(torch.device("cuda:0"))
        final = final_flat.view(B, C, int(nx_z), int(nx_x), int(nx_y)).to(torch.device("cuda:0"))
        
        final = torch.cat(final.unbind(dim=2), 1).to(torch.device("cuda:0"))
        
        return final

        # get tensors from the same voxel next to each other
        # ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
        #     + geom_feats[:, 1] * (self.nx[2] * B)\
        #     + geom_feats[:, 2] * B\
        #     + geom_feats[:, 3]
        # sorts = ranks.argsort()
        # x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # # cumsum trick
        # if not self.use_quickcumsum:
        #     x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        # else:
        #     x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        # Z = int(self.nx[2].item())
        # X = int(self.nx[0].item())
        # Y = int(self.nx[1].item())
        # K = Z * X * Y
        # final_flat = torch.zeros((B, C, K), device=x.device)
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        
        # linear_idx = (
        #     geom_feats[:, 3] * (self.nx[2] * self.nx[0] * self.nx[1]) +
        #     geom_feats[:, 2] * (self.nx[0] * self.nx[1]) +
        #     geom_feats[:, 0] * self.nx[1] +
        #     geom_feats[:, 1]
        # )  # shape: [N_kept]

        # # 배치별 누적
        # for b in range(B):
        #     pos = torch.nonzero(geom_feats[:, 3] == b, as_tuple=False).squeeze(1)  # [M]
        #     if pos.numel() == 0:
        #         continue
        #     idx_b = linear_idx.index_select(0, pos)              # [M]
        #     x_b = x.index_select(0, pos).transpose(0, 1)         # [C, M]
        
        # # 정적 그래프 변환을 위해 batch 1이라고 가정하고 for문 해제
        # # pos = torch.nonzero(geom_feats[:, 3] == 0, as_tuple=False).squeeze(1)
        # # idx_b = linear_idx.index_select(0, pos)
        # # x_b = x.index_select(0, pos).transpose(0, 1)
        
        # # final_flat[0].index_add_(1, idx_b, x_b)
        # S = torch.nn.functional.one_hot(idx_b.to(torch.int64), num_classes=K).to(x_b.dtype)  # [M, K]
        # final_flat[0] = x_b.matmul(S)  # [C, M] @ [M, K] -> [C, K]

        # final = final_flat.view(B, C, Z, X, Y)
        # final = torch.cat(final.unbind(dim=2), 1)

        # return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        print(x.shape, rots.shape, trans.shape, intrins.shape, post_rots.shape, post_trans.shape)
        x = x.to(torch.device("cuda:0"))
        rots = rots.to(torch.device("cuda:0"))
        trans = trans.to(torch.device("cuda:0"))
        intrins = intrins.to(torch.device("cuda:0"))
        post_rots = post_rots.to(torch.device("cuda:0"))
        post_trans = post_trans.to(torch.device("cuda:0"))
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)


# """
# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
# Authors: Jonah Philion and Sanja Fidler
# """

# import torch
# from torch import nn
# from efficientnet_pytorch import EfficientNet
# from torchvision.models.resnet import resnet18
# import numpy as np

# from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super().__init__()

#         self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
#                               align_corners=True)

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x1, x2):
#         x1 = x1
#         x2 = x2
#         x1 = self.up(x1)
#         x1 = torch.cat([x2, x1], dim=1)
#         return self.conv(x1)


# class CamEncode(nn.Module):
#     def __init__(self, D, C, downsample):
#         super(CamEncode, self).__init__()
#         self.D = D
#         self.C = C

#         self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

#         self.up1 = Up(320+112, 512)
#         self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

#     def get_depth_dist(self, x, eps=1e-20):
#         x = x
#         return x.softmax(dim=1)

#     def get_depth_feat(self, x):
#         x = x
#         x = self.get_eff_depth(x)
#         # Depth
#         x = self.depthnet(x)

#         depth = self.get_depth_dist(x[:, :self.D])
#         new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

#         new_x = new_x

#         return depth, new_x

#     def get_eff_depth(self, x):
#         # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
#         endpoints = dict()

#         # Stem
#         x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
#         prev_x = x

#         # Blocks
#         for idx, block in enumerate(self.trunk._blocks):
#             drop_connect_rate = self.trunk._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
#             x = block(x, drop_connect_rate=drop_connect_rate)
#             if prev_x.size(2) > x.size(2):
#                 endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
#             prev_x = x

#         # Head
#         endpoints['reduction_{}'.format(len(endpoints)+1)] = x
#         x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
#         return x

#     def forward(self, x):
#         depth, x = self.get_depth_feat(x)

#         return x


# class BevEncode(nn.Module):
#     def __init__(self, inC, outC):
#         super(BevEncode, self).__init__()

#         trunk = resnet18(pretrained=False, zero_init_residual=True)
#         self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = trunk.bn1
#         self.relu = trunk.relu

#         self.layer1 = trunk.layer1
#         self.layer2 = trunk.layer2
#         self.layer3 = trunk.layer3

#         self.up1 = Up(64+256, 256, scale_factor=4)
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear',
#                               align_corners=True),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, outC, kernel_size=1, padding=0),
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x1 = self.layer1(x)
#         x = self.layer2(x1)
#         x = self.layer3(x)

#         x = self.up1(x, x1)
#         x = self.up2(x)

#         return x


# class LiftSplatShoot(nn.Module):
#     def __init__(self, grid_conf, data_aug_conf, outC):
#         super(LiftSplatShoot, self).__init__()
#         self.grid_conf = grid_conf
#         self.data_aug_conf = data_aug_conf

#         dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
#                                               self.grid_conf['ybound'],
#                                               self.grid_conf['zbound'],
#                                               )
#         self.dx = nn.Parameter(dx, requires_grad=False)
#         self.bx = nn.Parameter(bx, requires_grad=False)
#         self.nx = nn.Parameter(nx, requires_grad=False)

#         self.downsample = 16
#         self.camC = 64
#         self.frustum = self.create_frustum()
#         self.D, _, _, _ = self.frustum.shape
#         self.camencode = CamEncode(self.D, self.camC, self.downsample)
#         self.bevencode = BevEncode(inC=self.camC, outC=outC)

#         # toggle using QuickCumsum vs. autograd
#         # self.use_quickcumsum = True
#         self.use_quickcumsum = False
        
#     # for onnx export
#     def safe_inverse_3x3(self, M, eps=1e-8):
#         a,b,c = M[...,0,0], M[...,0,1], M[...,0,2]
#         d,e,f = M[...,1,0], M[...,1,1], M[...,1,2]
#         g,h,i = M[...,2,0], M[...,2,1], M[...,2,2]
#         A = e*i - f*h
#         B = -(d*i - f*g)
#         C = d*h - e*g
#         D = -(b*i - c*h)
#         E = a*i - c*g
#         F = -(a*h - b*g)
#         G = b*f - c*e
#         H = -(a*f - c*d)
#         I = a*e - b*d
#         det = a*A + b*B + c*C
#         det = det.clamp(min=eps)
#         inv = torch.stack([
#             torch.stack([A, D, G], -1),
#             torch.stack([B, E, H], -1),
#             torch.stack([C, F, I], -1)
#         ], -2)
#         return (inv / det.unsqueeze(-1).unsqueeze(-1))
        
#     def create_frustum(self):
#         # make grid in image plane
#         ogfH, ogfW = self.data_aug_conf['final_dim']
#         fH, fW = ogfH // self.downsample, ogfW // self.downsample
#         ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
#         D, _, _ = ds.shape
#         xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
#         ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

#         # D x H x W x 3
#         frustum = torch.stack((xs, ys, ds), -1)
#         return nn.Parameter(frustum, requires_grad=False)

#     def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
#         """Determine the (x,y,z) locations (in the ego frame)
#         of the points in the point cloud.
#         Returns B x N x D x H/downsample x W/downsample x 3
#         """
#         B, N, _ = trans.shape

#         # undo post-transformation
#         # B x N x D x H x W x 3
#         points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
#         points = points
#         # points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

#         post_rots_inv = self.safe_inverse_3x3(post_rots)
#         points = post_rots_inv.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
#         # cam_to_ego
#         points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
#                             points[:, :, :, :, :, 2:3]
#                             ), 5)
#         # combine = rots.matmul(torch.inverse(intrins))
#         intrins_inv = self.safe_inverse_3x3(intrins)
#         combine = rots.matmul(intrins_inv)
        
#         points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
#         points += trans.view(B, N, 1, 1, 1, 3)

#         return points

#     def get_cam_feats(self, x):
#         """Return B x N x D x H/downsample x W/downsample x C
#         """
#         B, N, C, imH, imW = x.shape

#         x = x.view(B*N, C, imH, imW)
#         x = self.camencode(x)
#         x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
#         x = x.permute(0, 1, 3, 4, 5, 2)

#         return x

#     def voxel_pooling(self, geom_feats, x):
#         B, N, D, H, W, C = x.shape
#         Nprime = B*N*D*H*W

#         # flatten x
#         x = x.reshape(Nprime, C)

#         # flatten indices
#         geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
#         geom_feats = geom_feats.view(Nprime, 3)
#         batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
#                              device=x.device, dtype=torch.long) for ix in range(B)])
#         geom_feats = torch.cat((geom_feats, batch_ix), 1)

#         # filter out points that are outside box
#         # kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
#         #     & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
#         #     & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
#         nx_val = self.nx.to(x.device)
#         zeros = torch.zeros_like(geom_feats[:, 0])
#         kept_x = (geom_feats[:, 0] >= zeros) & (geom_feats[:, 0] < nx_val[0])
#         kept_y = (geom_feats[:, 1] >= zeros) & (geom_feats[:, 1] < nx_val[1])
#         kept_z = (geom_feats[:, 2] >= zeros) & (geom_feats[:, 2] < nx_val[2])

#         kept = kept_x & kept_y & kept_z
#         x = x[kept]
#         geom_feats = geom_feats[kept]
        
#         # Use a fully ONNX-compatible method (One-Hot + MatMul) for aggregation.
#         nx_x, nx_y, nx_z = nx_val[0], nx_val[1], nx_val[2]
#         K = int(nx_x * nx_y * nx_z)

#         ranks = geom_feats[:, 3] * K + \
#                 geom_feats[:, 2] * (nx_x * nx_y) + \
#                 geom_feats[:, 0] * nx_y + \
#                 geom_feats[:, 1]
#         ranks = ranks.long()

#         one_hot = torch.nn.functional.one_hot(ranks, num_classes=B * K).float()
        
#         final_very_flat = (one_hot.transpose(0, 1) @ x)

#         final_flat = final_very_flat.view(B, K, C).permute(0, 2, 1)
#         final = final_flat.view(B, C, int(nx_z), int(nx_x), int(nx_y))
        
#         final = torch.cat(final.unbind(dim=2), 1)
        
#         return final

#         # get tensors from the same voxel next to each other
#         # ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
#         #     + geom_feats[:, 1] * (self.nx[2] * B)\
#         #     + geom_feats[:, 2] * B\
#         #     + geom_feats[:, 3]
#         # sorts = ranks.argsort()
#         # x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

#         # # cumsum trick
#         # if not self.use_quickcumsum:
#         #     x, geom_feats = cumsum_trick(x, geom_feats, ranks)
#         # else:
#         #     x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

#         # griddify (B x C x Z x X x Y)
#         # Z = int(self.nx[2].item())
#         # X = int(self.nx[0].item())
#         # Y = int(self.nx[1].item())
#         # K = Z * X * Y
#         # final_flat = torch.zeros((B, C, K), device=x.device)
#         # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        
#         # linear_idx = (
#         #     geom_feats[:, 3] * (self.nx[2] * self.nx[0] * self.nx[1]) +
#         #     geom_feats[:, 2] * (self.nx[0] * self.nx[1]) +
#         #     geom_feats[:, 0] * self.nx[1] +
#         #     geom_feats[:, 1]
#         # )  # shape: [N_kept]

#         # # 배치별 누적
#         # for b in range(B):
#         #     pos = torch.nonzero(geom_feats[:, 3] == b, as_tuple=False).squeeze(1)  # [M]
#         #     if pos.numel() == 0:
#         #         continue
#         #     idx_b = linear_idx.index_select(0, pos)              # [M]
#         #     x_b = x.index_select(0, pos).transpose(0, 1)         # [C, M]
        
#         # # 정적 그래프 변환을 위해 batch 1이라고 가정하고 for문 해제
#         # # pos = torch.nonzero(geom_feats[:, 3] == 0, as_tuple=False).squeeze(1)
#         # # idx_b = linear_idx.index_select(0, pos)
#         # # x_b = x.index_select(0, pos).transpose(0, 1)
        
#         # # final_flat[0].index_add_(1, idx_b, x_b)
#         # S = torch.nn.functional.one_hot(idx_b.to(torch.int64), num_classes=K).to(x_b.dtype)  # [M, K]
#         # final_flat[0] = x_b.matmul(S)  # [C, M] @ [M, K] -> [C, K]

#         # final = final_flat.view(B, C, Z, X, Y)
#         # final = torch.cat(final.unbind(dim=2), 1)

#         # return final

#     def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
#         geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
#         x = self.get_cam_feats(x)

#         x = self.voxel_pooling(geom, x)

#         return x

#     def forward(self, x, rots, trans, intrins, post_rots, post_trans):
#         # print(x.shape, rots.shape, trans.shape, intrins.shape, post_rots.shape, post_trans.shape)
#         x = x
#         rots = rots
#         trans = trans
#         intrins = intrins
#         post_rots = post_rots
#         post_trans = post_trans
#         x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
#         x = self.bevencode(x)
#         return x


# def compile_model(grid_conf, data_aug_conf, outC):
#     return LiftSplatShoot(grid_conf, data_aug_conf, outC)
