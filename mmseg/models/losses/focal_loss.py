# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmseg.registry import MODELS
from .utils import weight_reduce_loss


# This method is used when cuda is not available
def py_sigmoid_focal_loss(pred,
                          target,
                          one_hot_target=None,
                          weight=None,
                          gamma=2.0,
                          alpha=0.5,
                          class_weight=None,
                          valid_mask=None,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction with
            shape (N, C)
        one_hot_target (None): Placeholder. It should be None.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float | list[float], optional): A balanced form for Focal Loss.
            Defaults to 0.5.
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        valid_mask (torch.Tensor, optional): A mask uses 1 to mark the valid
            samples and uses 0 to mark the ignored samples. Default: None.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    if isinstance(alpha, list):
        alpha = pred.new_tensor(alpha)
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * one_minus_pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    final_weight = torch.ones(1, pred.size(1)).type_as(loss)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       one_hot_target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.5,
                       class_weight=None,
                       valid_mask=None,
                       reduction='mean',
                       avg_factor=None):
    r"""CUDA 版本 `Focal Loss <https://arxiv.org/abs/1708.02002>`_ 的包装器。

    Args:
        pred (torch.Tensor): 预测值，形状为 (N, C)，其中 C 为类别数。
        target (torch.Tensor): 预测值的学习标签，形状应为 (N,)。
        one_hot_target (torch.Tensor): 使用 one-hot 编码的学习标签，形状为 (N, C)。
        weight (torch.Tensor, optional): 样本的损失权重。
        gamma (float, optional): 用于计算调制因子的 gamma 参数，默认值为 2.0。
        alpha (float | list[float], optional): Focal Loss 的平衡系数，默认值为 0.5。
        class_weight (list[float], optional): 每个类别的权重，默认为 None。
        valid_mask (torch.Tensor, optional): 标记有效样本的掩码，用 1 表示有效样本，用 0 表示忽略的样本。默认为 None。
        reduction (str, optional): 用于将损失减少为标量的方法，默认为 'mean'。选项为 "none", "mean" 和 "sum"。
        avg_factor (int, optional): 用于平均损失的平均因子，默认为 None。

    Returns:
        torch.Tensor: 计算得到的损失值
    """
    # Function.apply 不接受关键字参数，因此装饰器 "weighted_loss" 不适用。
    final_weight = torch.ones(1, pred.size(1)).type_as(pred)
    if isinstance(alpha, list):
        # _sigmoid_focal_loss 不接受列表类型的 alpha 参数。
        # 因此，如果给定了一个列表，我们将 alpha 设置为 0.5，表示前景类和背景类的权重相等。
        # 通过将损失乘以 2，可以消除将 alpha 设置为 0.5 的效果。
        # 列表类型的 alpha 参数将在后处理过程中用于调整损失。
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                   gamma, 0.5, None, 'none') * 2
        alpha = pred.new_tensor(alpha)
        # 更新最终的权重，考虑到 alpha 和 one-hot 目标
        final_weight = final_weight * (
                alpha * one_hot_target + (1 - alpha) * (1 - one_hot_target))
    else:
        # 如果 alpha 不是列表，则直接计算损失
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                   gamma, alpha, None, 'none')

    # 如果提供了样本权重，则调整权重形状并应用到最终权重
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # 在大多数情况下，weight 的形状为 (N,)，即它没有第二个维度 num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight

    # 如果提供了类别权重，则应用到最终权重
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)

    # 如果提供了有效掩码，则应用到最终权重
    if valid_mask is not None:
        final_weight = final_weight * valid_mask

    # 通过加权缩减方法计算最终的损失
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)

    return loss


@MODELS.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.5,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_focal'):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional):
                是否使用 Sigmoid 激活函数来处理预测值，默认为 True。
            gamma (float, optional):
                用于计算调制因子的 gamma 参数，默认值为 2.0。
            alpha (float | list[float], optional):
                Focal Loss 的平衡形式。默认为 0.5。当提供列表时，列表的长度应等于类别数。请注
                意，此参数不是类别权重，而是二分类问题的权重。此二分类问题将属于一个类别的像素视
                为前景，将其他像素视为背景，列表中的每个元素都是相应前景类别的权重。alpha 的值
                或 alpha 的每个元素应为区间 [0, 1] 中的浮点数。如果要指定类别权重，请使用
                `class_weight` 参数。
            reduction (str, optional):
                用于减少损失的方式，默认为 'mean'。可选项为 "none", "mean", "sum"。
            class_weight (list[float], optional): 每个类别的权重，默认为 None。
            loss_weight (float, optional): 损失的权重，默认为 1.0。
            loss_name (str, optional):
                损失项的名称。如果想将此损失项包含在反向传播图中，`loss_` 必须是名称的前缀，
                默认为 'loss_focal'。
        """
        super().__init__()
        assert use_sigmoid is True, \
            'AssertionError: Only sigmoid focal loss supported now.'
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        assert isinstance(class_weight, list) or class_weight is None, \
            'AssertionError: class_weight must be None or of type list'

        self.use_sigmoid = use_sigmoid  # 是否使用 Sigmoid 激活
        self.gamma = gamma  # Focal Loss 中的 gamma 参数
        self.alpha = alpha  # Focal Loss 中的 alpha 参数
        self.reduction = reduction  # 损失函数的 reduction 方法
        self.class_weight = class_weight  # 类别权重
        self.loss_weight = loss_weight  # 损失的权重
        self._loss_name = loss_name  # 损失名称

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """前向传播函数。

        Args:
            pred (torch.Tensor): 预测值，形状为 (N, C)，其中 C 为类别数，
                或 (N, C, d_1, d_2, ..., d_K)，K≥1 时为 K 维损失。
            target (torch.Tensor): 真实标签。若包含类别索引，形状为 (N)，
                其中 0≤targets[i]≤C−1，或 (N, d_1, d_2, ..., d_K)，K≥1 时为 K 维损失。
                若包含类别概率，形状与输入相同。
            weight (torch.Tensor, optional): 每个预测值的损失权重，默认为 None。
            avg_factor (int, optional): 用于平均损失的平均因子，默认为 None。
            reduction_override (str, optional): 用于覆盖损失的 reduction 方法，可选项为 "none", "mean", "sum"。
            ignore_index (int, optional): 要忽略的标签索引，默认为 255。

        Returns:
            torch.Tensor: 计算得到的损失值
        """
        assert isinstance(ignore_index, int), \
            'ignore_index 必须为 int 类型'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "reduction 应为 'none', 'mean' 或 'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
            "pred 的形状与 target 的形状不匹配"

        original_shape = pred.shape

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            # target with shape [B, C, d_1, d_2, ...]
            # transform it's shape into [N, C]
            # [B, C, d_1, d_2, ...] -> [C, B, d_1, d_2, ..., d_k]
            target = target.transpose(0, 1)
            # [C, B, d_1, d_2, ..., d_k] -> [C, N]
            target = target.reshape(target.size(0), -1)
            # [C, N] -> [N, C]
            target = target.transpose(0, 1).contiguous()
        else:
            # target with shape [B, d_1, d_2, ...]
            # transform it's shape into [N, ]
            target = target.view(-1).contiguous()
            valid_mask = (target != ignore_index).view(-1, 1)
            # avoid raising error when using F.one_hot()
            target = torch.where(target == ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            if torch.cuda.is_available() and pred.is_cuda:
                if target.dim() == 1:
                    one_hot_target = F.one_hot(
                        target, num_classes=num_classes + 1)
                    if num_classes == 1:
                        # 如果只有一个类，则处理为二分类问题
                        one_hot_target = one_hot_target[:, 1]
                        target = 1 - target
                    else:
                        one_hot_target = one_hot_target[:, :num_classes]
                else:
                    one_hot_target = target
                    target = target.argmax(dim=1)
                    valid_mask = (target != ignore_index).view(-1, 1)
                calculate_loss_func = sigmoid_focal_loss
            else:
                one_hot_target = None
                if target.dim() == 1:
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    if num_classes == 1:
                        target = target[:, 1]
                    else:
                        target = target[:, num_classes]
                else:
                    valid_mask = (target.argmax(dim=1) != ignore_index).view(
                        -1, 1)
                calculate_loss_func = py_sigmoid_focal_loss

            # 计算损失
            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                one_hot_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                class_weight=self.class_weight,
                valid_mask=valid_mask,
                reduction=reduction,
                avg_factor=avg_factor)

            if reduction == 'none':
                # [N, C] -> [C, N]
                loss_cls = loss_cls.transpose(0, 1)
                # [C, N] -> [C, B, d1, d2, ...]
                # original_shape: [B, C, d1, d2, ...]
                loss_cls = loss_cls.reshape(original_shape[1],
                                            original_shape[0],
                                            *original_shape[2:])
                # [C, B, d1, d2, ...] -> [B, C, d1, d2, ...]
                loss_cls = loss_cls.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@MODELS.register_module()
class CombinedFocalIoULoss(nn.Module):
    def __init__(self,
                 iou_loss_weight=2.0,
                 loss_name='loss_FocalIoU',
                 # Focal Loss参数
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.5,
                 class_weight=None,
                 loss_weight=1.0,
                 reduction='mean',
                 ):
        """
        Focal Loss 和 IoU Loss 的组合损失函数。

        Args:
            focal_loss_params (dict, optional): Focal Loss 的参数字典。
            ciou_loss_weight (float, optional): CIoU Loss 的权重。默认值为 1.0。
            reduction (str, optional): 用于将损失减少为标量的方法。默认为 'mean'。选项为 "none", "mean" 和 "sum"。
            ignore_index (int, optional): 要忽略的标签索引。默认值为 255。
        """
        super().__init__()
        assert use_sigmoid is True, \
            'AssertionError: Only sigmoid focal loss supported now.'
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        assert isinstance(class_weight, list) or class_weight is None, \
            'AssertionError: class_weight must be None or of type list'

        # 初始化 Focal Loss
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight

        # 初始化CIoU
        self.iou_loss_weight = iou_loss_weight
        self._loss_name = loss_name

    def forward(
            self,
            pred,
            target,
            # Focal Loss 参数
            weight=None,
            avg_factor=None,
            reduction_override=None,
            ignore_index=255,
            **kwargs
    ):
        assert isinstance(ignore_index, int), \
            'ignore_index 必须为 int 类型'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "reduction 应为 'none', 'mean' 或 'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
            "pred 的形状与 target 的形状不匹配"

        original_shape = pred.shape

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            # target with shape [B, C, d_1, d_2, ...]
            # transform it's shape into [N, C]
            # [B, C, d_1, d_2, ...] -> [C, B, d_1, d_2, ..., d_k]
            target = target.transpose(0, 1)
            # [C, B, d_1, d_2, ..., d_k] -> [C, N]
            target = target.reshape(target.size(0), -1)
            # [C, N] -> [N, C]
            target = target.transpose(0, 1).contiguous()
        else:
            # target with shape [B, d_1, d_2, ...]
            # transform it's shape into [N, ]
            target = target.view(-1).contiguous()
            valid_mask = (target != ignore_index).view(-1, 1)
            # avoid raising error when using F.one_hot()
            target = torch.where(target == ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.use_sigmoid:
            num_classes = pred.size(1)
            if torch.cuda.is_available() and pred.is_cuda:
                if target.dim() == 1:
                    one_hot_target = F.one_hot(
                        target, num_classes=num_classes + 1)
                    if num_classes == 1:
                        # 如果只有一个类，则处理为二分类问题
                        one_hot_target = one_hot_target[:, 1]
                        target = 1 - target
                    else:
                        one_hot_target = one_hot_target[:, :num_classes]
                else:
                    one_hot_target = target
                    target = target.argmax(dim=1)
                    valid_mask = (target != ignore_index).view(-1, 1)
                calculate_loss_func = sigmoid_focal_loss

            # 计算损失
            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                one_hot_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                class_weight=self.class_weight,
                valid_mask=valid_mask,
                reduction=reduction,
                avg_factor=avg_factor)

        # 计算 CIoU Loss
        iou_loss = self.iou_loss(pred, target)

        # 组合两个损失
        loss_cls = loss_cls + self.iou_loss_weight * iou_loss

        if reduction == 'none':
            # [N, C] -> [C, N]
            loss_cls = loss_cls.transpose(0, 1)
            # [C, N] -> [C, B, d1, d2, ...]
            # original_shape: [B, C, d1, d2, ...]
            loss_cls = loss_cls.reshape(original_shape[1],
                                        original_shape[0],
                                        *original_shape[2:])
            # [C, B, d1, d2, ...] -> [B, C, d1, d2, ...]
            loss_cls = loss_cls.transpose(0, 1).contiguous()

        return loss_cls

    def iou_loss(self, pred, target,num_classes=2, ignore_index=255):
        """
        计算 soft IoU Loss。

        Args:
            pred (torch.Tensor): 预测值，形状为 (N, C)。
            target (torch.Tensor): 真实标签，形状为 (N) 。
            ignore_index (int, optional): 要忽略的标签索引。默认值为 255。

        Returns:
            torch.Tensor: 计算得到的 CIoU 损失值。
        """
        eps = 1e-6

        # 获取预测的类别概率
        preds = torch.softmax(pred, dim=1)

        # 创建忽略索引的掩码
        valid_mask = (target != ignore_index)

        # 初始化 IoU 损失
        total_iou = 0.0
        num_valid_classes = 0

        for cls in range(num_classes):
            # 获取当前类别的预测和真实标签
            pred = preds[:, cls]  # 形状: (N, H, W)
            target = (target == cls).float()  # 形状: (N, H, W)

            # 应用有效掩码
            pred = pred * valid_mask
            target = target * valid_mask

            # 计算交集和并集
            intersection = torch.sum(pred * target)
            union = torch.sum(pred + target) - intersection

            # 仅当 union > 0 时计算 IoU
            if union > 0:
                iou = (intersection + eps) / (union + eps)
                total_iou += (1 - iou)
                num_valid_classes += 1

            # 如果存在有效的类别，则计算平均损失，否则损失为0
        if num_valid_classes == 0:
            return torch.tensor(0.0, requires_grad=True)
        else:
            return total_iou / num_valid_classes

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name