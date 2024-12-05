import colored as cl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mnetwork import TrigonometricActivation, mThresholdActivation


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        # print(self.mask.shape, similarity_matrix.shape)
        denominator = self.mask.to(similarity_matrix) * torch.exp(
            similarity_matrix / self.temperature
        )

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss


class reweightMSELoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction="mean"):
        super().__init__()
        self._weight = weight
        self._size_average = size_average
        self._reduce = reduce
        self._reduction = reduction

    def forward(self, input, target):
        device = input.device
        loss = (input - target).to(device)
        loss = torch.square(loss)
        loss = torch.mean(loss, dim=0)

        # reweight
        if self._weight is not None:
            # clone weight
            weight = self._weight.clone().detach().to(device)
            loss = loss * weight

        loss = torch.mean(loss)

        return loss


class TeacherNet(nn.Module):
    def __init__(self, in_dim=1200, out_dim=1024, hidden_dim=2048):
        super().__init__()

        # standard MLP network

        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            #
            nn.Dropout(0.1),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            #
            nn.Dropout(0.1),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            #
        )

        # self.projection = nn.Sequential(
        #     nn.Linear(in_features=in_dim, out_features=out_dim),
        #     # nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        # )
        # self.net_1 = nn.Sequential(
        #     nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        # )
        # self.net_2 = nn.Sequential(
        #     nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        # )
        # self.net_3 = nn.Sequential(
        #     nn.Linear(in_features=hidden_dim, out_features=out_dim),
        #     nn.ReLU(),
        # )

        # generate middle net for the teacher net
        #  decrease the hidden_dim by half util the out_dim

        # self.net = [
        #     # nn.BatchNorm1d(in_dim),
        #     nn.Linear(in_features=in_dim, out_features=hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True),
        # ]
        # while hidden_dim > np.square(out_dim):
        #     self.net.append(
        #         nn.Linear(
        #             in_features=hidden_dim,
        #             out_features=np.round(np.sqrt(hidden_dim)).astype(int),
        #         )
        #     )
        #     hidden_dim = np.round(np.sqrt(hidden_dim)).astype(int)
        #     self.net.append(nn.ReLU(inplace=True))

        # self.net.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))
        # self.net.append(nn.ReLU(inplace=True))

        # self.net = nn.Sequential(*self.net)

    def forward(self, x):
        out = self.mlp(x)

        # out = self.projection(x)
        # out = self.net_1(out) + out
        # out = self.net_2(out) + out
        # out = self.net_3(out)

        return out


class StudentNet(nn.Module):
    def __init__(self, in_dim=1200, out_dim=1024, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            #
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(out_dim),
            #
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

        # self.projection = nn.Sequential(
        #     nn.Linear(in_features=in_dim, out_features=hidden_dim),
        # )
        # self.net_1 = nn.Sequential(
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        #     nn.Dropout(0.1),
        #     nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim),
        #     # nn.BatchNorm1d(hidden_dim),
        #     nn.PReLU(),
        # )
        # self.net_2 = nn.Sequential(
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        #     nn.Dropout(0.1),
        #     nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim),
        #     # nn.BatchNorm1d(hidden_dim),
        #     nn.PReLU(),
        # )
        # self.net_3 = nn.Sequential(
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        #     nn.Dropout(0.1),
        #     nn.Linear(in_features=hidden_dim // 2, out_features=out_dim),
        #     nn.PReLU(),
        # )

    def forward(self, x):
        out = self.net(x)

        # out = self.projection(x)
        # out = self.net_1(out) + out
        # out = self.net_2(out) + out
        # out = self.net_3(out)

        return out


# TODO:need to modify for the pytorch lightning, this is the minumum version base on the original paper
# Improving Self-Supervised Learning by Characterizing Idealized Representations
# https://arxiv.org/pdf/2209.06235.pdf
class CISSL(nn.Module):
    def __init__(self, z_dim=1200, proj_dim=1024, hidden_dim=2048):
        super().__init__()
        # self.encoder = resnet()  # to define
        # teacher projection should be as expressive as possible
        self.teacher_proj = TeacherNet(
            z_dim, proj_dim, hidden_dim=hidden_dim
        )  # to define
        # student projection should be linear ( note : BN is linear )
        self.student_proj = StudentNet(z_dim, proj_dim, hidden_dim=hidden_dim)

    def loss(self, x1, x2, temp=0.07):
        # x1, x2 = batch
        bs, device = x1.size(0), x1.device
        # logits shape : [2*bs , 2*bs ]. Normalizes for cosine sim.
        # z = self.encoder(torch.cat([x1, x2], dim=0))
        z = torch.cat([x1, x2], dim=0)
        z_student = F.normalize(self.student_proj(z), dim=1, p=2)
        z_teacher = F.normalize(self.teacher_proj(z), dim=1, p=2)
        logits = z_student @ z_teacher.T / temp
        # there are two positives for each example x1: x1 and x2
        # note : SimCLR removes x1 -x1 as those are typically equal .
        # But not for CISSL due to asymmetric proj heads = >
        # CE between predicted proba and 0.5 for each positive
        log_q = logits.log_softmax(-1)
        select_pos = torch.eye(bs, device=device).bool().repeat(2, 2)
        CE = -log_q[select_pos].view(bs * 2, 2).sum(1) / 2
        return CE.mean()

    def forward(self, x1, x2):
        # z1, z2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = x1, x2
        loss = self.loss(z1, z2)
        assert loss >= 0, f"Loss is negative : {loss}"
        return loss


# TODO:need to modify for the pytorch lightning, this is the minumum version base on the original paper
# Improving Self-Supervised Learning by Characterizing Idealized Representations
# https://arxiv.org/pdf/2209.06235.pdf
class DISSL(nn.Module):
    def __init__(self, z_dim=1024, n_equiv=16384, hidden_dim=2048):
        super().__init__()
        # self.encoder = resnet()  # to define
        # teacher projection should be as expressive as possible
        self.teacher_proj = TeacherNet(z_dim, n_equiv, hidden_dim=hidden_dim)
        # student projection should be same architeture as probe
        # self.student_proj = nn.Linear(z_dim, n_equiv)
        self.student_proj = StudentNet(z_dim, n_equiv, hidden_dim=hidden_dim)

    def loss(self, x1, x2):
        # z1, z2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = x1, x2
        return (self.asym_loss(z1, z2) + self.asym_loss(z2, z1)) / 2

    def asym_loss(self, z1, z2, lambd=2.3, beta=0.8, temp=0.5):
        temp = 0.0001
        logits_t1 = self.teacher_proj(z1) / temp
        logits_t2 = self.teacher_proj(z2) / temp
        logits_s = self.student_proj(z2)

        q_Mlx = torch.distributions.Categorical(logits=logits_t1)  # q(\ hat {M}|X)
        # MAXIMALITY . -H[\ hat {M}]
        mxml = -torch.distributions.Categorical(probs=q_Mlx.probs.mean(0)).entropy()
        # INVARIANCE and DETERMINISM . E_ {q(M|X)}[log q(M|\ tilde {X})]
        det_inv = (q_Mlx.probs * logits_t2.log_softmax(-1)).sum(-1)
        # DISTILLATION . E_ {q(M|X)}[log s(M|\ tilde {X})]
        dstl = (q_Mlx.probs * logits_s.log_softmax(-1)).sum(-1)
        return lambd * mxml - beta * det_inv.mean() - dstl.mean()

    def forward(self, z1, z2):
        return self.loss(z1, z2)


class simpleDistanceLoss(nn.Module):
    def forward(self, x, y):
        target = torch.linalg.norm(y[:, :2], dim=1)

        return {
            "loss": F.mse_loss(
                x.reshape(-1, 1) * 1000, target=1000 * target.reshape(-1, 1)
            ),
            "distance": ((x.reshape(-1, 1) - target.reshape(-1, 1)) * 100).abs().mean(),
            "error_distribution": ((x.reshape(-1, 1) - target.reshape(-1, 1)) * 100),
        }


class DistanceLoss(nn.Module):
    def __init__(self, ratio=1.0):
        super().__init__()
        self._ratio = ratio
        # self.filter = mThresholdActivation(threshold=0.1 / np.sqrt(3), value=0.0)
        # self.filter = TrigonometricActivation()

    def forward(self, x, y):
        local_x = torch.clone(x)
        local_y = torch.clone(y)
        # check nan in local_x and local_y
        assert torch.isnan(local_x).sum() == 0
        assert torch.isnan(local_y).sum() == 0

        # local_y = self.filter(local_y)
        local_y = local_y[:, :2]

        # sum_local_x = torch.sum(local_x, dim=1) * 100
        # sum_local_y = torch.sum(local_y, dim=1) * 100

        local_x = local_x * 1_00
        local_y = local_y * 1_00

        local_x.requires_grad_(True)

        """
        distance
        """
        distance = torch.abs((local_x - local_y))
        distance = torch.linalg.norm(distance, dim=1, keepdim=True)
        distance = torch.sum(distance, dim=1)
        distance = torch.mean(distance)

        """
        angle
        """
        angle_label_XY = torch.atan2(local_y[:, 1], local_y[:, 0])
        angle_pred_XY = torch.atan2(local_x[:, 1], local_x[:, 0])
        angle_XY = torch.abs(angle_label_XY - angle_pred_XY)
        # angle_pred_XY.requires_grad_(True)
        # angle_XY = torch.abs(angle_label_XY - angle_pred_XY)
        # angle_XY = torch.fmod(angle_XY, 2 * torch.pi)
        # angle_pred_XY.requires_grad_(True)
        # over = torch.where(angle_XY > torch.pi)
        # angle_XY[over] = 2 * torch.pi - angle_XY[over]
        # angle_XY = angle_XY.reshape(-1, 1)

        # over90 = (angle_XY > (torch.pi / 2)).float().requires_grad_(True)
        # over45 = (angle_XY > (torch.pi / 4)).float().requires_grad_(True)
        # over30 = (angle_XY > (torch.pi / 6)).float().requires_grad_(True)
        # over15 = (angle_XY > (torch.pi / 12)).float().requires_grad_(True)
        # over10 = (angle_XY > (torch.pi / 18)).float().requires_grad_(True)
        # over5 = (angle_XY > (torch.pi / 36)).float().requires_grad_(True)
        # over3 = (angle_XY > (torch.pi / 60)).float().requires_grad_(True)
        # # stack all over
        # over = torch.stack(
        #     [
        #         over90,
        #         over45,
        #         over30,
        #         over15,
        #         over10,
        #         over5,
        #         over3,
        #     ],
        #     dim=1,
        # )
        over = angle_XY / torch.pi

        """
        sin cos error
        """
        norm_x = torch.linalg.norm(local_x, dim=-1, keepdim=True)
        norm_y = torch.linalg.norm(local_y, dim=-1, keepdim=True)

        norm_x = norm_x.clone()
        norm_y = norm_y.clone()
        norm_x[norm_x == 0] = 1.0
        norm_y[norm_y == 0] = 1.0

        tri_x = local_x / norm_x
        tri_y = local_y / norm_y

        sin_error = torch.abs(tri_x[:, 0] - tri_y[:, 0])
        cos_error = torch.abs(tri_x[:, 1] - tri_y[:, 1])

        """
        direction
        """
        magicPoint = torch.logical_and(local_x == 0, local_y == 0)
        wrong_direction = (local_x * local_y) <= 0
        wrong_direction[magicPoint] = False
        wrong_sector = torch.logical_or(
            wrong_direction[:, 0], wrong_direction[:, 1]
        ).float()
        wrong_direction = wrong_direction.float()

        """
        average X and Y
        """
        avg_x = torch.mean(local_x[:, 0])
        avg_y = torch.mean(local_x[:, 1])
        ratio = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
        ratio = torch.tensor(ratio, device=local_x.device)
        ratio = ratio / torch.sum(ratio)

        """
        True Loss
        """

        loss = (
            F.mse_loss(
                local_x,
                local_y,
            )
            + F.huber_loss(
                tri_x,
                tri_y,
            )
            + F.binary_cross_entropy_with_logits(
                over,
                torch.zeros_like(over),
            )
        )

        return {
            "loss": loss,
            "distance": distance.mean(),
            "angle_XY": angle_XY.mean() * 180 / np.pi,  # radian to degree
            "sector": wrong_sector.mean(),
            "avg_x": avg_x,
            "avg_y": avg_y,
        }


class AngleLoss(nn.Module):
    def __init__(self, ratio, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ratio = ratio

    def forward(self, x, y):
        angle = torch.arctan2(y[:, 1], y[:, 0]) - x
        angle = torch.square(angle)
        angle = torch.mean(angle)

        return angle


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, output, target):
        output_norm = output / (output.norm(p=2, dim=1, keepdim=True) + self.eps)
        target_norm = target / (target.norm(p=2, dim=1, keepdim=True) + self.eps)
        cos_sim = torch.sum(output_norm * target_norm, dim=1)
        return 1 - cos_sim.mean()


class PairLoss(nn.Module):
    def __init__(self):
        super(PairLoss, self).__init__()
        self.filter = mThresholdActivation(threshold=0.1 / np.sqrt(3), value=0.0)

    def forward(self, x, y):
        local_y = self.filter(y)
        # x_norm = torch.linalg.norm(x[:, :2], dim=1, keepdim=True)
        y_norm = torch.linalg.norm(local_y[:, :2], dim=1, keepdim=True)
        y_norm[y_norm == 0] = 1.0

        # x_cossin = x[:, :2]
        # y_cossin = y[:, :2] / y_norm

        # angle = torch.arctan2(x[:, 1], x[:, 0])
        angle = x[:, 0]
        strength = x[:, -1]

        # unit convert
        strength = (strength * 100).reshape(-1, 1)
        angle = (angle * 180 / np.pi).reshape(-1, 1)

        target_strength = (torch.linalg.norm(local_y[:, :2], dim=1) * 100).reshape(
            -1, 1
        )
        target_angle = (
            torch.atan2(local_y[:, 1], local_y[:, 0]) * 180 / torch.pi
        ).reshape(-1, 1)

        assert strength.shape == target_strength.shape
        assert angle.shape == target_angle.shape

        distance_loss = F.smooth_l1_loss(
            strength,
            target_strength,
        )
        diff = (angle - target_angle).abs()

        diff = diff % 360
        over = torch.where(diff > 180)
        diff[over] = 360 - diff[over]
        diff = diff.reshape(-1, 1)

        # over90 = (diff > 90).float()
        # over45 = (diff > 45).float()
        # over30 = (diff > 30).float()
        # over15 = (diff > 15).float()
        # over10 = (diff > 10).float()
        # over5 = (diff > 5).float()
        # over3 = (diff > 3).float()
        # over1 = (diff > 1).float()

        # over = torch.stack(
        #     [
        #         over90,
        #         over45,
        #         over30,
        #         over15,
        #         over10,
        #         over5,
        #         over3,
        #         over1,
        #     ],
        #     dim=1,
        # )
        over = diff / 180

        angle_loss = F.l1_loss(
            angle,
            target_angle,
        )

        # angle_loss = F.smooth_l1_loss(
        #     angle,
        #     target_angle,
        # ) + F.binary_cross_entropy_with_logits(
        #     over,
        #     torch.zeros_like(over),
        # )

        # angle_loss = torch.log2((angle - target_angle).abs() + 1).mean()

        # vec = torch.stack(
        #     [
        #         torch.cos(x[:, 1]) * x[:, 0],
        #         torch.sin(x[:, 1]) * x[:, 0],
        #     ],
        #     dim=1,
        # )

        # mix = F.mse_loss(
        #     vec * 100,
        #     y[:, :2] * 100,
        # )

        loss = distance_loss + angle_loss  # + mix

        return {
            "loss": loss,
            "angle_loss": angle_loss,
            "vel_loss": distance_loss,
            "angle": (diff).abs().mean(),
            "distance": (strength - target_strength).abs().mean(),
        }


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2.0 * x) - np.log(2.0)

    return torch.mean(_log_cosh(y_pred - y_true))


class timeWiseLoss(nn.Module):
    # def __init__(self, ratio=1.0):
    #     super().__init__()
    #     self._ratio = ratio
    def forward(self, x, y):
        target = y * 100
        single_v_x = torch.sum(x, dim=1) * 0.01
        single_v_y = torch.sum(target, dim=1) * 0.01

        angle_pred = torch.atan2(single_v_x[:, 1], single_v_x[:, 0])
        angle_target = torch.atan2(single_v_y[:, 1], single_v_y[:, 0])

        angle_XY = torch.abs(angle_target - angle_pred)
        angle_XY = torch.fmod(angle_XY, 2 * torch.pi)
        over = torch.where(angle_XY > torch.pi)
        angle_XY[over] = 2 * torch.pi - angle_XY[over]
        angle_XY = angle_XY.reshape(-1, 1)

        loss = F.mse_loss(x, target) + F.mse_loss(single_v_x, single_v_y)

        return {
            "loss": loss,
            "angle_XY": angle_XY.mean() * 180 / np.pi,  # radian to degree
            "distance": torch.linalg.norm(single_v_x - single_v_y, dim=1).mean() * 100,
        }
