from __future__ import annotations

from typing import Type

from torch import nn

from monai.networks.layers import Norm
from umei.utils import ChannelFirst, ChannelLast

__all__ = ['LayerNormNd']

class LayerNormNd(nn.Sequential):
    def __init__(self, num_channels: int):
        super().__init__(
            ChannelLast(),
            nn.LayerNorm(num_channels),
            ChannelFirst(),
        )

@Norm.factory_function("layernd")
def layer_factory(_dim) -> Type[LayerNormNd]:
    return LayerNormNd
