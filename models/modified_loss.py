import torch
import torch.nn as nn


class GeoAwareMaskLoss(nn.Module):
    """
    融合几何约束与光影鲁棒的 Mask 损失函数
    在标准的分割损失（如 Dice Loss 或 BCE Loss）基础上，增加甘蔗长宽比的惩罚项。
    """

    def __init__(self, alpha=1.0, beta=0.5, target_ratio=5.0):
        super(GeoAwareMaskLoss, self).__init__()
        self.alpha = alpha  # 基础 Mask Loss 权重
        self.beta = beta  # 几何约束权重
        self.target_ratio = target_ratio  # 预设的甘蔗合理长宽比先验
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_masks, target_masks, pred_bboxes):
        # 1. 计算基础的 BCE Loss
        base_loss = self.bce(pred_masks, target_masks)

        # 2. 计算几何惩罚项 (基于长宽比)
        # 假设 pred_bboxes 格式为 [x_center, y_center, width, height]
        w = pred_bboxes[:, 2]
        h = pred_bboxes[:, 3]

        # 避免除以 0
        w = torch.clamp(w, min=1e-6)
        current_ratios = h / w

        # 计算当前预测框比例与目标先验比例的均方误差
        ratio_penalty = torch.mean((current_ratios - self.target_ratio) ** 2)

        # 3. 融合损失
        total_loss = (self.alpha * base_loss) + (self.beta * ratio_penalty)
        return total_loss