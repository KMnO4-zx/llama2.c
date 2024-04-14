#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   llama2_model.py
@Time    :   2024/04/14 22:26:35
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   部分代码借鉴llama2.c仓库代码
'''

import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    # 自定义超参数
    dim: int = 288  # 模型维度
    n_layers: int = 6  # Transformer层数
    n_heads: int = 6  # 注意力机制的头数
    n_kv_heads: Optional[int] = 6  # 键/值头数，如果未指定，则默认为n_heads
    vocab_size: int = 32000  # 词汇表大小
    hidden_dim: Optional[int] = None  # 隐藏层维度，如果未指定，则使用其他规则确定
    multiple_of: int = 32  # MLP隐藏层大小是这个数的倍数
    norm_eps: float = 1e-5  # 归一化层的epsilon值
    max_seq_len: int = 256  # 最大序列长度
    dropout: float = 0.0  # 丢弃率


class LLaMA2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps是为了防止除以0的情况
        self.eps = eps
        # weight是一个可学习的参数，全部初始化为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算RMSNorm的核心部分
        # x.pow(2).mean(-1, keepdim=True)计算了输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # forward函数是模型的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

