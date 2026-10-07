import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.head import Detect


class DualHeadYOLO(nn.Module):
    def __init__(
        self,
        weights="yolo11m.pt",
        person_nc=1,
        apparatus_nc=29,
    ):
        super().__init__()

        # =====================================
        # 1. 加载原始 YOLO11m
        # =====================================

        base_model = YOLO(weights).model


        # =====================================
        # 2. 获取完整 Ultralytics 训练配置
        # =====================================

        # v8DetectionLoss 会读取：
        #
        # model.args.box
        # model.args.cls
        # model.args.dfl
        #
        # checkpoint 中 base_model.args 并不一定
        # 包含这些训练 loss 超参数。
        #
        # 所以这里直接取得 Ultralytics
        # 完整默认配置。

        self.args = get_cfg()


        # =====================================
        # 3. 共享 Backbone + Neck
        # 去掉原模型最后一个 Detect
        # =====================================

        self.shared_layers = nn.ModuleList(
            list(base_model.model[:-1])
        )


        # YOLO 网络需要保存的中间层编号
        self.save = set(base_model.save)


        # Detect 使用的三个尺度
        # P3 / P4 / P5
        self.feature_ids = (
            16,
            19,
            22,
        )


        # YOLO11m Neck 输出通道
        feature_channels = (
            256,
            512,
            512,
        )


        # =====================================
        # 4. 两个独立检测头
        # =====================================

        # 人员检测：1 类
        self.person_head = Detect(
            nc=person_nc,
            ch=feature_channels,
        )


        # 化学器皿检测：29 类
        self.apparatus_head = Detect(
            nc=apparatus_nc,
            ch=feature_channels,
        )


        # =====================================
        # 5. 设置 stride
        # =====================================

        original_stride = (
            base_model.model[-1]
            .stride
            .detach()
            .clone()
        )


        self.person_head.stride = (
            original_stride.clone()
        )

        self.apparatus_head.stride = (
            original_stride.clone()
        )


        # 初始化 Detect bias
        self.person_head.bias_init()
        self.apparatus_head.bias_init()


    # =====================================
    # 共享 Backbone + Neck
    # =====================================

    def forward_shared(self, x):

        outputs = []

        for m in self.shared_layers:

            # 处理 YOLO skip connection
            if m.f != -1:

                if isinstance(m.f, int):

                    x = outputs[m.f]

                else:

                    x = [
                        x if j == -1
                        else outputs[j]

                        for j in m.f
                    ]

            x = m(x)

            outputs.append(
                x if m.i in self.save
                else None
            )


        # Detect 所需的 P3 / P4 / P5
        features = [
            outputs[i]
            for i in self.feature_ids
        ]

        return features


    # =====================================
    # 运行指定检测头
    # =====================================

    def forward_head(
        self,
        features,
        task,
    ):

        if task == "person":

            return self.person_head(
                features
            )


        if task == "apparatus":

            return self.apparatus_head(
                features
            )


        raise ValueError(
            f"未知任务: {task}"
        )


    # =====================================
    # 同时运行两个检测头
    # =====================================

    def forward(self, x):

        features = self.forward_shared(x)

        person_output = (
            self.person_head(features)
        )

        apparatus_output = (
            self.apparatus_head(features)
        )

        return {
            "person": person_output,
            "apparatus": apparatus_output,
            "features": features,
        }


# =========================================
# 基础测试
# =========================================

if __name__ == "__main__":

    model = DualHeadYOLO()

    model.train()

    x = torch.randn(
        1,
        3,
        640,
        640,
    )

    outputs = model(x)


    print(
        "========== LOSS HYPERPARAMETERS =========="
    )

    print(
        "box:",
        model.args.box
    )

    print(
        "cls:",
        model.args.cls
    )

    print(
        "dfl:",
        model.args.dfl
    )


    print(
        "\n========== PERSON =========="
    )

    print(
        "boxes:",
        outputs["person"]["boxes"].shape
    )

    print(
        "scores:",
        outputs["person"]["scores"].shape
    )


    print(
        "\n========== APPARATUS =========="
    )

    print(
        "boxes:",
        outputs["apparatus"]["boxes"].shape
    )

    print(
        "scores:",
        outputs["apparatus"]["scores"].shape
    )


    print(
        "\n========== SUCCESS =========="
    )

    print(
        "dual head forward success!"
    )