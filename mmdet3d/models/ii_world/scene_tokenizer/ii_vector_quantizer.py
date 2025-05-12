""" adapted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from mmdet.models.builder import BACKBONES
from mmcv.runner import BaseModule

@BACKBONES.register_module()
class IntraInterVectorQuantizer(BaseModule):
    def __init__(self, n_e, e_dim, beta, z_channels, remap=None, unknown_index="random", recover_stage=3, recover_time=3,
                 sane_index_shape=False, legacy=True, use_voxel=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.recover_stage = recover_stage
        self.recover_time = recover_time
        self.recover_scale_conv = nn.ModuleList()
        self.recover_time_conv = nn.ModuleList()
        for i in range(self.recover_stage):
            self.recover_scale_conv.append(nn.Conv2d(self.e_dim, self.e_dim, 1))

        for i in range(self.recover_time):
            self.recover_time_conv.append(nn.Conv2d(self.e_dim, self.e_dim, 1))

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        conv_class = torch.nn.Conv3d if use_voxel else torch.nn.Conv2d
        self.quant_conv = conv_class(z_channels, self.e_dim, 1)
        self.post_quant_conv = conv_class(self.e_dim, z_channels, 1)

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, curr_bev, sampled_bev, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        curr_bev = self.quant_conv(curr_bev)
        # z: [bs, c, w, h]
        B, C, W, H = curr_bev.shape
        shapes = []
        for i in range(self.recover_stage):
            w = W // (2 ** (self.recover_stage - i - 1))
            h = H // (2 ** (self.recover_stage - i - 1))
            shapes.append((w, h))

        mean_vq_loss = 0
        z_no_grad = curr_bev
        z_rest = z_no_grad.clone()
        z_hat = torch.zeros_like(z_rest)
        encoding_list = []
        for i in range(self.recover_stage):
            rest_i_shaep = shapes[i]
            z_rest_i = F.interpolate(z_rest, rest_i_shaep, mode='bilinear', align_corners=False)

            z_q_i, loss_i, (perplexity, min_encodings, min_encoding_indices) = self.forward_quantizer(z_rest_i, temp,
                                                                                                      rescale_logits,
                                                                                                      return_logits,
                                                                                                      is_voxel)

            z_q_i = F.interpolate(z_q_i, (W, H), mode='bilinear', align_corners=False)
            z_hat = z_hat + self.recover_scale_conv[i](z_q_i)
            z_rest = z_rest - z_q_i

            mean_vq_loss += loss_i
            encoding_list.append(min_encoding_indices)

        mean_vq_loss *= 1 / self.recover_stage

        for i in range(self.recover_time):
            z_rest_i = z_rest + sampled_bev[:, i]

            z_q_i, _, _ = self.forward_quantizer(z_rest_i, temp, rescale_logits, return_logits, is_voxel)

            z_hat = z_hat + self.recover_time_conv[i](z_q_i)

            z_rest = z_rest - z_q_i

        z_q = self.post_quant_conv(z_hat)
        return z_q, mean_vq_loss, (perplexity, min_encodings, min_encoding_indices)

    def forward_quantizer(self, z, temp=None, rescale_logits=False, return_logits=False, is_voxel=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"

        # reshape z -> (batch, height, width, channel) and flatten
        if not is_voxel:
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
        else:
            z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if not is_voxel:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        else:
            z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            if not is_voxel:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3])
            else:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
