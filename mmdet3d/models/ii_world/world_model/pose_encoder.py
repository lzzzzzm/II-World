import torch
import torch.nn as nn
from mmdet.models import HEADS

from mmcv.runner import BaseModule, force_fp32

def quaternion_to_rotation_matrix(quaternion):
    # Quterion to Rotation Matrix
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    batch_size = quaternion.size(0)
    R = torch.zeros((batch_size, 3, 3), device=quaternion.device, dtype=quaternion.dtype)

    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)

    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - w * x)

    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    return R


def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v


def stereographic_project(a):
    dim = a.shape[1]
    a = normalize_vector(a)
    out = a[:, 0:dim - 1] / (1 - a[:, dim - 1])
    return out


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

def quart_to_rpy(qua):
    x, y, z, w = qua[:, 0], qua[:, 1], qua[:, 2], qua[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(2 * (w * y - x * z))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

    return roll, pitch, yaw


@HEADS.register_module()
class PoseEncoder(BaseModule):
    def __init__(
            self,
            embed_dims=128,
            traj_axis=2,
            history_frame_number=5,
            positional_encoding=None,
            ortho6d=False,
    ):
        super(PoseEncoder, self).__init__()

        # in_channels = (2+4+3+3) * history_frame_number
        in_channels = (2+3+3+4)* history_frame_number

        self.plan_input = nn.Sequential(
            nn.Linear(in_channels, embed_dims),
            nn.ReLU(True),
            nn.Linear(embed_dims, embed_dims),
        )

        self.traj_axis = traj_axis
        self.history_frame_number = history_frame_number

        self.ego_yaw_prev = None
        self.ego_pos_prev = None
        self.ego_width, self.ego_length = 1.85, 4.084


    def get_curr_to_futu(self, pred_rotation, pred_delta_translation, curr_rotation, curr_translations,
                         curr_ego_to_global, start_of_sequence):
        bs = pred_rotation.shape[0]

        _, _, ego_yaw = quart_to_rpy(pred_rotation)

        # make rotations to quaternion
        pred_rotation = quaternion_to_rotation_matrix(pred_rotation)

        # Change delta translation to absolute translation
        pred_translation = torch.zeros(bs, 3, device=pred_rotation.device, dtype=pred_rotation.dtype)
        pred_translation[:, :2] = pred_delta_translation[:, :2] + curr_translations[:, :2]
        pred_translation[:, 2] = curr_translations[:, 2]

        # make transformation matrix
        pred_ego_to_global = torch.zeros(bs, 4, 4, device=pred_rotation.device, dtype=pred_rotation.dtype)
        pred_ego_to_global[:, :3, :3] = pred_rotation
        pred_ego_to_global[:, :3, 3] = pred_translation
        pred_ego_to_global[:, 3, 3] = 1

        # make curr_to_next transformation
        curr_to_future = pred_ego_to_global.inverse() @ curr_ego_to_global
        curr_to_future[:, 3, :3] = torch.zeros(bs, 3, device=pred_rotation.device, dtype=pred_rotation.dtype)

        if self.ego_yaw_prev is None:
            _, _, ego_yaw_prev = quart_to_rpy(curr_rotation)
            self.ego_yaw_prev = ego_yaw_prev
            self.ego_pos_prev = curr_translations

        # Get ego_lcf_feat
        if start_of_sequence.sum() > 0:
            _, _, ego_yaw_prev = quart_to_rpy(curr_rotation)
            self.ego_yaw_prev[start_of_sequence] = ego_yaw_prev[start_of_sequence]
            self.ego_pos_prev[start_of_sequence] = curr_translations[start_of_sequence]

        ego_lcf_feat = torch.zeros(bs, 3, device=pred_rotation.device, dtype=pred_rotation.dtype)
        ego_w = (ego_yaw - self.ego_yaw_prev) / 0.5
        ego_v = torch.linalg.norm(pred_translation[:2] - self.ego_pos_prev[:2]) / 0.5
        ego_vx, ego_vy = ego_v * torch.cos(ego_yaw + torch.pi / 2), ego_v * torch.sin(ego_yaw + torch.pi / 2)
        ego_lcf_feat[:, 0] = ego_w
        ego_lcf_feat[:, 1] = ego_vx
        ego_lcf_feat[:, 2] = ego_vy

        # update ego_yaw_prev and ego_pos_prev
        self.ego_yaw_prev = ego_yaw.clone()
        self.ego_pos_prev = pred_translation.clone()

        return curr_to_future.detach().clone().contiguous(), pred_ego_to_global.detach().clone().contiguous(), ego_lcf_feat.clone().detach()

    def get_ego_feat(self, pred_rotation, pred_translation, curr_rotation, curr_translations, start_of_sequence):
        """

        Args:
            pred_rotation: quaternion
            pred_delta_translation: absolute translation
            curr_rotation: current quaternion
            curr_translations: absolute translation
            start_of_sequence:

        Returns:

        """
        bs = pred_rotation.shape[0]

        _, _, ego_yaw = quart_to_rpy(pred_rotation)

        # make rotations to quaternion
        pred_rotation = quaternion_to_rotation_matrix(pred_rotation)

        if self.ego_yaw_prev is None:
            _, _, ego_yaw_prev = quart_to_rpy(curr_rotation)
            self.ego_yaw_prev = ego_yaw_prev
            self.ego_pos_prev = curr_translations

        # Get ego_lcf_feat
        if start_of_sequence.sum() > 0:
            _, _, ego_yaw_prev = quart_to_rpy(curr_rotation)
            self.ego_yaw_prev[start_of_sequence] = ego_yaw_prev[start_of_sequence]
            self.ego_pos_prev[start_of_sequence] = curr_translations[start_of_sequence]

        ego_lcf_feat = torch.zeros(bs, 3, device=pred_rotation.device, dtype=pred_rotation.dtype)
        ego_w = (ego_yaw - self.ego_yaw_prev) / 0.5
        ego_v = torch.linalg.norm(pred_translation[:2] - self.ego_pos_prev[:2]) / 0.5
        ego_vx, ego_vy = ego_v * torch.cos(ego_yaw + torch.pi / 2), ego_v * torch.sin(ego_yaw + torch.pi / 2)
        ego_lcf_feat[:, 0] = ego_w
        ego_lcf_feat[:, 1] = ego_vx
        ego_lcf_feat[:, 2] = ego_vy

        # update ego_yaw_prev and ego_pos_prev
        self.ego_yaw_prev = ego_yaw.clone()
        self.ego_pos_prev = pred_translation.clone()

        return ego_lcf_feat.clone().detach()


    @force_fp32()
    def forward_encoder(self,
                        history_delta_translation,
                        history_relative_rotation,
                        history_ego_mode,
                        history_ego_lcf_feat,
                ):
        # Process scene latent
        # current latent: bs, c, w, h

        bs, f = history_delta_translation.shape[0], history_delta_translation.shape[1]

        plan_query = torch.cat([history_delta_translation, history_relative_rotation, history_ego_mode, history_ego_lcf_feat], dim=-1)
        plan_query = plan_query.reshape(bs, -1)

        plan_query = self.plan_input(plan_query)
        plan_query = plan_query.unsqueeze(1)

        # plan query : [bs, 1, embed_dims]
        return plan_query
