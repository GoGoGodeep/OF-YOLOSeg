# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .typing_utils import SampleList


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def stack_batch(inputs: List[torch.Tensor],
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """
    堆叠多个输入以形成一个批次，并填充图像和 gt_sem_segs 以达到最大形状，使用右下填充模式

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
            输入的多个张量，每个都是一个CHW（通道、高度、宽度）的3D张量。
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
            数据样本的列表，通常包括 `gt_sem_seg` 等信息。
        size (tuple, optional): Fixed padding size.
            固定的填充尺寸。
        size_divisor (int, optional): The divisor of padded size.
            填充尺寸的除数。
        pad_val (int, float): The padding value. Defaults to 0.
            填充值。默认为0。
        seg_pad_val (int, float): The padding value. Defaults to 255.
            分割填充值。默认为255。

    Returns:
       Tensor: The 4D-tensor.
           返回的4D张量。
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
           填充后的数据样本列表。
    """
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    # 初始化存储填充后的图像张量和数据样本的列表
    padded_inputs = []
    padded_samples = []

    # 获取每个输入图像的尺寸（高度和宽度）
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]

    # 计算所有输入图像的最大尺寸
    max_size = np.stack(inputs_sizes).max(0)

    # 如果 size_divisor 不为空且大于1，则调整 max_size 以满足整除要求
    if size_divisor is not None and size_divisor > 1:
        # 最后两个维度是高度和宽度，都需要满足整除要求
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]

        # 如果 size 参数不为空，计算所需的填充尺寸
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        # 如果 size_divisor 参数不为空，计算所需的填充尺寸
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # 对图像张量进行填充
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)

        # 对数据样本进行填充
        if data_samples is not None:
            data_sample = data_samples[i]
            pad_shape = None

            # 填充 gt_sem_seg
            if 'gt_sem_seg' in data_sample:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(
                    gt_sem_seg, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_sem_seg.shape

            # 填充 gt_edge_map
            if 'gt_edge_map' in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_edge_map.shape

            # 填充 gt_depth_map
            if 'gt_depth_map' in data_sample:
                gt_depth_map = data_sample.gt_depth_map.data
                del data_sample.gt_depth_map.data
                data_sample.gt_depth_map.data = F.pad(
                    gt_depth_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_depth_map.shape

            # 更新数据样本的元信息
            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': pad_shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(
                    img_padding_size=padding_size,
                    pad_shape=pad_img.shape[-2:]))

    # # 调试信息：打印每个张量的尺寸以确保填充正确
    # for idx, padded_input in enumerate(padded_inputs):
    #     print(f"Tensor {idx} size after padding: {padded_input.shape}")

    # 检查所有张量的尺寸是否一致
    tensor_shapes = [tensor.shape for tensor in padded_inputs]
    assert len(set(tensor_shapes)) == 1, \
        f"All tensors must have the same shape, but got {tensor_shapes}"

    # 返回填充后的图像张量和数据样本列表
    return torch.stack(padded_inputs, dim=0), padded_samples
