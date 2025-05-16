# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)


# class InnerSioULoss(nn.Module):
#     def __init__(self, eps=1e-7, ratio=0.2):
#         super(InnerSioULoss, self).__init__()
#         self.eps = eps
#         self.ratio = ratio

#     def forward(self, pred_bboxes, target_bboxes, fg_mask):
#         """
#         Compute the InnerSIoU loss.

#         Parameters:
#         - pred_bboxes: [N, 4] - (x_min, y_min, x_max, y_max)
#         - target_bboxes: [N, 4] - (x_min, y_min, x_max, y_max)
#         - fg_mask: [N] - foreground mask

#         Returns:
#         - loss: scalar tensor
#         """
#         pred = pred_bboxes[fg_mask]
#         target = target_bboxes[fg_mask]

#         # Compute widths, heights, centers
#         xgt_c = (target[:, 0] + target[:, 2]) / 2
#         ygt_c = (target[:, 1] + target[:, 3]) / 2
#         wgt = target[:, 2] - target[:, 0]
#         hgt = target[:, 3] - target[:, 1]

#         xc = (pred[:, 0] + pred[:, 2]) / 2
#         yc = (pred[:, 1] + pred[:, 3]) / 2
#         w = pred[:, 2] - pred[:, 0]
#         h = pred[:, 3] - pred[:, 1]

#         # Ground truth box corners with ratio
#         bgt_l = xgt_c - (wgt * self.ratio) / 2
#         bgt_r = xgt_c + (wgt * self.ratio) / 2
#         bgt_t = ygt_c - (hgt * self.ratio) / 2
#         bgt_b = ygt_c + (hgt * self.ratio) / 2

#         # Predicted box corners with ratio
#         bl = xc - (w * self.ratio) / 2
#         br = xc + (w * self.ratio) / 2
#         bt = yc - (h * self.ratio) / 2
#         bb = yc + (h * self.ratio) / 2

#         # Intersection
#         inter_w = torch.clamp(torch.min(bgt_r, br) - torch.max(bgt_l, bl), min=0)
#         inter_h = torch.clamp(torch.min(bgt_b, bb) - torch.max(bgt_t, bt), min=0)
#         inter = inter_w * inter_h

#         # Union
#         union = (wgt * hgt * self.ratio ** 2) + (w * h * self.ratio ** 2) - inter + self.eps

#         iou_inner = inter / union  # Inner IoU

#         # Original SIoU (just IoU for simplicity here — replace with actual SIoU if needed)
#         # Standard IoU
#         x1 = torch.max(pred[:, 0], target[:, 0])
#         y1 = torch.max(pred[:, 1], target[:, 1])
#         x2 = torch.min(pred[:, 2], target[:, 2])
#         y2 = torch.min(pred[:, 3], target[:, 3])

#         inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
#         pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
#         target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
#         union_area = pred_area + target_area - inter_area + self.eps
#         iou = inter_area / union_area  # Standard IoU

#         # Loss = SIoU + IoU - Inner IoU
#         # Replace iou with actual SIoU loss if using full SIoU definition
#         loss = 1 - iou + iou - iou_inner  # => 1 - iou_inner

#         return loss.mean()
    



class InnerSioULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(InnerSioULoss, self).__init__()
        self.eps = eps

    def forward(self, pred_bboxes, target_bboxes, fg_mask):
        """
        Compute the InnerSioU loss.

        Parameters:
        - pred_bboxes: Predicted bounding boxes (tensor of shape [N, 4])
        - target_bboxes: Ground truth bounding boxes (tensor of shape [N, 4])
        - fg_mask: Binary mask to select foreground (tensor of shape [N])

        Returns:
        - loss: Computed InnerSioU loss
        """
        # Select the foreground bounding boxes
        pred_bboxes_fg = pred_bboxes[fg_mask]
        target_bboxes_fg = target_bboxes[fg_mask]

        # Calculate intersection area
        inter_xmin = torch.max(pred_bboxes_fg[:, 0], target_bboxes_fg[:, 0])
        inter_ymin = torch.max(pred_bboxes_fg[:, 1], target_bboxes_fg[:, 1])
        inter_xmax = torch.min(pred_bboxes_fg[:, 2], target_bboxes_fg[:, 2])
        inter_ymax = torch.min(pred_bboxes_fg[:, 3], target_bboxes_fg[:, 3])

        inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

        # Calculate union area
        pred_area = (pred_bboxes_fg[:, 2] - pred_bboxes_fg[:, 0]) * (pred_bboxes_fg[:, 3] - pred_bboxes_fg[:, 1])
        target_area = (target_bboxes_fg[:, 2] - target_bboxes_fg[:, 0]) * (target_bboxes_fg[:, 3] - target_bboxes_fg[:, 1])

        union_area = pred_area + target_area - inter_area

        # Compute the inner IoU
        iou = inter_area / (union_area + self.eps)

        # InnerSioU focuses on reducing the error between the intersection area of the predicted and target bounding boxes
        loss = 1 - iou  # The closer iou is to 1, the smaller the loss

        # Return the average loss for the foreground bounding boxes
        return loss.mean()