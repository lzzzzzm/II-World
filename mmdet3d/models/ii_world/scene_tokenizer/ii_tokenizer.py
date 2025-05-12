# Modified from OccWorld
import cv2
import copy
import time

import mmcv
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet.models import DETECTORS

from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models import builder

from mmdet3d.models.losses.lovasz_softmax import lovasz_softmax

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

@DETECTORS.register_module()
class IISceneTokenizer(CenterPoint):
    def __init__(self,
                 encoder=None,
                 decoder=None,
                 vq=None,
                 num_classes=18,
                 class_embeds_dim=8,
                 frame_number=2,
                 vq_channel=None,
                 # loss_cfg
                 grid_config=None,
                 empty_idx=None,
                 use_class_weights=False,
                 class_weights=None,
                 lovasz_loss=None,
                 focal_loss=None,
                 embed_loss_weight=None,
                 save_results=True,
                 **kwargs):
        super(IISceneTokenizer, self).__init__(**kwargs)
        # ---------------------- init params ------------------------------
        self.num_classes = num_classes
        self.class_embeds_dim = class_embeds_dim

        # ---------------------- init Model ------------------------------
        self.encoder = builder.build_backbone(encoder)
        self.vq = builder.build_backbone(vq)
        self.decoder = builder.build_backbone(decoder)
        # Time module
        self.history_bev = None
        self.bev_aug = None
        self.frame_number = frame_number
        self.vq_channel = vq_channel
        x_config, y_config, z_config = grid_config['x'], grid_config['y'], grid_config['z']
        dx, bx, nx = gen_dx_bx(x_config, y_config, z_config)
        self.dx, self.bx, self.nx = dx, bx, nx
        # Embedding
        self.class_embeds = nn.Embedding(num_classes, class_embeds_dim)
        # Others
        self.save_results = save_results
        if self.save_results:
            mmcv.mkdir_or_exist('save_dir')
        # Losses
        self.empty_idx = empty_idx
        self.use_class_weights = use_class_weights
        self.class_weights = torch.tensor(np.array(class_weights), dtype=torch.float32, device='cuda')
        self.focal_loss = builder.build_loss(focal_loss)
        self.embed_loss_weight = embed_loss_weight

    def reconstruct_loss(self, pred, targ):
        # pred: [bs, T, W, H, Z, C]
        # targ: [bs, T, W, H, Z]
        # Change pred to [bs*T, c, w, h, z]
        bs, T, W, H, Z, C = pred.shape
        pred = pred.reshape(bs*T, W, H, Z, C).permute(0, 4, 1, 2, 3)
        targ = targ.reshape(bs*T, W, H, Z)

        loss_dict = dict()
        loss_reconstruct = self.focal_loss(pred, targ,
                                           None if not self.use_class_weights else self.class_weights,
                                           ignore_index=255)  # self-reconstruction loss
        loss_lovasz = lovasz_softmax(torch.softmax(pred, dim=1), targ, ignore=255)

        loss_dict['recon_loss'] = loss_reconstruct + loss_lovasz

        return loss_dict

    def generate_grid(self, curr_bev):
        n, c_, z, h, w = curr_bev.shape
        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h,w, z, 4, 1)
        return grid

    def generate_feat2bev(self, grid, dx, bx):
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = dx[0]
        feat2bev[1, 1] = dx[1]
        feat2bev[2, 2] = dx[2]
        feat2bev[0, 3] = bx[0] - dx[0] / 2.
        feat2bev[1, 3] = bx[1] - dx[1] / 2.
        feat2bev[2, 3] = bx[2] - dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        return feat2bev

    def align_bev(self, curr_bev, img_metas):
        # z_sampled: [bs, c, w, h]
        curr_bev = curr_bev.permute(0, 1, 3, 2).unsqueeze(2)   # change to [bs, c, h, w], z=1
        bs, c, z, h, w = curr_bev.shape
        # prepare
        # check if start of sequence
        start_of_sequence = np.array([img_meta['start_of_sequence'] for img_meta in img_metas])
        curr_to_prev_ego_rt = torch.stack([torch.tensor(img_meta['curr_to_prev_ego_rt'], device=curr_bev.device) for img_meta in img_metas])
        bev_aug = torch.stack([img_meta['bda_mat'].to(curr_bev.device) for img_meta in img_metas])

        if self.history_bev is None:
            self.history_bev = curr_bev.repeat(1, self.frame_number, 1, 1, 1).clone()
            self.bev_aug = bev_aug.clone()

        if start_of_sequence.sum() > 0:
            self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.frame_number, 1, 1, 1)
            self.bev_aug[start_of_sequence] = bev_aug[start_of_sequence]

        self.history_bev = self.history_bev.detach()

        tmp_bev = self.history_bev

        # align different time step
        # Generate grid
        grid = self.generate_grid(curr_bev)
        feat2bev = self.generate_feat2bev(grid, self.dx, self.bx)

        rt_flow = (torch.inverse(feat2bev) @ self.bev_aug @ curr_to_prev_ego_rt @ torch.inverse(bev_aug) @ feat2bev)
        grid = rt_flow.view(bs, 1, 1, 1, 4, 4) @ grid

        normalize_factor = torch.tensor([w - 1.0, h - 1.0, 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0  # grid order is x, y, z
        grid[..., 2] = 0.0

        sampled_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4),  align_corners=True, mode='bilinear')

        # Time fusion
        bev_cat = torch.cat([curr_bev, sampled_bev], dim=1)
        self.history_bev = bev_cat[:, :-self.vq_channel, ...].detach().clone()

        sampled_bev = sampled_bev.reshape(bs, self.frame_number, c, z, h, w).permute(0, 1, 2, 3, 5, 4).squeeze(3)  # change to w, h and suqeeze z-axis

        return sampled_bev.clone()

    def forward_encoder(self, voxel_semantics):
        # voxel_semantics: [bs, T, W, H, Z]
        BS, T, H, W, Z = voxel_semantics.shape
        voxel_semantics = self.class_embeds(voxel_semantics)
        voxel_semantics = voxel_semantics.reshape(BS*T, H, W, Z * self.class_embeds_dim).permute(0, 3, 1, 2)
        z, shapes = self.encoder(voxel_semantics)
        return z, shapes

    def forward_decoder(self, z, shapes, input_shape):
        # z:[bs, C, H, W], input_shape: original shape of voxel_semantics
        logits = self.decoder(z, shapes)

        bs, F, H, W, D = input_shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, D, self.class_embeds_dim)
        template = self.class_embeds.weight.T.unsqueeze(0)  # 1, expansion, cls
        similarity = torch.matmul(logits, template)  # -1, D, cls

        return similarity.reshape(bs, F, H, W, D, self.num_classes)

    def forward_test(self, voxel_semantics, img_metas, **kwargs):
        # 0. Prepare Input
        bs, t, w, h, d = voxel_semantics.shape

        # 2. Process current voxel semantics
        curr_bev, shapes = self.forward_encoder(voxel_semantics)

        # 3. Time fusion
        sampled_bev = self.align_bev(curr_bev, img_metas)

        # 4. vq
        z_sampled, loss, info = self.vq(curr_bev, sampled_bev, is_voxel=False)

        # 5. Process Decoder
        logits = self.forward_decoder(z_sampled, shapes, (bs, 1, w, h, d))

        # 6. Preprocess logits
        output_dict = dict()
        pred = logits.softmax(-1).argmax(-1).cpu().numpy()
        if self.save_results:
            # z_sampled: [bs, c, h, w]
            # rel_poses = np.array([img_meta['rel_poses'][0] for img_meta in img_metas])[0]
            # gt_mode = np.array([img_meta['gt_mode'] for img_meta in img_metas])[0]
            save_token = z_sampled[0].cpu().numpy()
            mmcv.mkdir_or_exist('data/nuscenes/save_dir/token_4f/{}'.format(img_metas[0]['scene_name']))
            np.savez('data/nuscenes/save_dir/token_4f/{}/{}.npz'.format(img_metas[0]['scene_name'], img_metas[0]['sample_idx']),token=save_token)

            # save pred
            # mmcv.mkdir_or_exist('save_dir/debug_8f/{}'.format(img_metas[0]['scene_name']))
            # np.savez('save_dir/debug_8f/{}/{}.npz'.format(img_metas[0]['scene_name'],img_metas[0]['sample_idx']), semantics=pred[0][0])

        output_dict['semantics'] = pred.astype(np.uint8)
        output_dict['index'] = [img_meta['index'] for img_meta in img_metas]
        return [output_dict]

    def forward_train(self, voxel_semantics, img_metas, **kwargs):
        # 0. Prepare Input
        bs, t, w, h, d = voxel_semantics.shape

        # 2. Process current voxel semantics
        curr_bev, shapes = self.forward_encoder(voxel_semantics)

        # 3. Time fusion
        sampled_bev = self.align_bev(curr_bev, img_metas)

        # 4. vq
        z_sampled, loss, info = self.vq(curr_bev, sampled_bev, is_voxel=False)

        # 5. Process Decoder
        logits = self.forward_decoder(z_sampled, shapes, (bs, 1, w, h, d))

        # 6. Compute Loss
        loss_dict = dict()
        loss_dict.update(self.reconstruct_loss(logits, voxel_semantics))
        loss_dict['embed_loss'] = self.embed_loss_weight * loss
        return loss_dict
