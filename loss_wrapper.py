import torch

from ultralytics.nn.modules.head import Detect


class LossModelWrapper(torch.nn.Module):
    """
    给 v8DetectionLoss 使用的包装器

    让官方 loss 认为：
    
    wrapper.model[-1]

    是一个 Detect 层
    """

    def __init__(self, head, args):
        super().__init__()

        self.model = torch.nn.ModuleList(
            [head]
        )

        self.args = args

        self.class_weights = None


    def forward(self, x):
        return self.model[-1](x)