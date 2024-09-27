import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLoULoss(nn.Module):
    def __init__(self, batch=32):
        super(SoftLoULoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        loss1 = self.bce_loss(pred, target)
        return loss + loss1


class SoftLoULoss1(nn.Module):
    def __init__(self, batch=32):
        super(SoftLoULoss1, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        loss1 = self.bce_loss(pred, target)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 平衡因子
        self.gamma = gamma  # 聚焦因子
        self.reduction = reduction

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        pt = torch.exp(-bce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss