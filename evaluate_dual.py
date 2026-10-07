import os
import cv2
import torch
import numpy as np

from tqdm import tqdm

from dual_head_model import DualHeadYOLO
from ultralytics.utils.nms import non_max_suppression


# ============================================================
# 基本配置
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model_path = os.environ.get("DUAL_EVAL_WEIGHTS", "checkpoints/dual_head_finetune_p4_a1_best.pt")

image_size = 640

# 为了计算完整 PR 曲线，AP 评估使用很低的置信度阈值
ap_conf_threshold = 0.001

# Precision / Recall 单独使用这个置信度阈值
pr_conf_threshold = 0.25

nms_iou_threshold = 0.45

iou_thresholds = np.arange(
    0.50,
    0.96,
    0.05
)


# ============================================================
# Apparatus 类别
# ============================================================

apparatus_names = [
    "Beaker",
    "Buchner_Funnel",
    "Burette_Stands",
    "Calorimeter",
    "Conical_Flask",
    "Funnel",
    "Glass_Rod",
    "Measuring_Cylinder",
    "Mechanical_Balance_Scale",
    "Nessler_Reagent_Bottle",
    "Pipette",
    "Porcelain_Mortar_Pestle",
    "Precision_Weight_Scale",
    "Reagent_Bottle",
    "Round_Bottom_Flask_Borosilicate_Glass_1_Neck",
    "Round_Bottom_Flask_Borosilicate_Glass_2_Neck",
    "Round_Bottom_Flask_Borosilicate_Glass_3_Neck",
    "Separating_Funnel",
    "Spirit_Lamp",
    "TestTube_Holder",
    "Test_Tube",
    "Volumetric_Flask",
    "Volumetric_Pipet",
    "Wash_Bottle",
    "Weighing_Bottle",
    "Hand",
    "Conical_Beaker",
    "Generic_Pipette",
    "Eggplant_Shaped_Flask",
]

person_names = [
    "Person"
]


# ============================================================
# IoU
# ============================================================

def box_iou(box1, box2):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    inter_w = max(
        0.0,
        x2 - x1
    )

    inter_h = max(
        0.0,
        y2 - y1
    )

    inter = (
        inter_w
        *
        inter_h
    )

    area1 = max(
        0.0,
        box1[2] - box1[0]
    ) * max(
        0.0,
        box1[3] - box1[1]
    )

    area2 = max(
        0.0,
        box2[2] - box2[0]
    ) * max(
        0.0,
        box2[3] - box2[1]
    )

    union = (
        area1
        +
        area2
        -
        inter
    )

    if union <= 0:
        return 0.0

    return inter / union


# ============================================================
# 读取 YOLO 标签
# ============================================================

def load_yolo_labels(
    label_path,
    width,
    height
):

    targets = []

    if not os.path.exists(
        label_path
    ):
        return targets

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                continue

            cls_id = int(
                float(values[0])
            )

            xc = (
                float(values[1])
                *
                width
            )

            yc = (
                float(values[2])
                *
                height
            )

            bw = (
                float(values[3])
                *
                width
            )

            bh = (
                float(values[4])
                *
                height
            )

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            targets.append(
                {
                    "cls": cls_id,
                    "box": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                }
            )

    return targets


# ============================================================
# AP 积分
# ============================================================

def compute_ap(
    recall,
    precision
):

    if len(recall) == 0:
        return 0.0

    mrec = np.concatenate(
        (
            [0.0],
            recall,
            [1.0]
        )
    )

    mpre = np.concatenate(
        (
            [1.0],
            precision,
            [0.0]
        )
    )

    # Precision envelope
    mpre = np.flip(
        np.maximum.accumulate(
            np.flip(mpre)
        )
    )

    # 使用较密集的 recall 网格积分
    x = np.linspace(
        0,
        1,
        101
    )

    ap = np.trapezoid(
        np.interp(
            x,
            mrec,
            mpre
        ),
        x
    )

    return float(ap)


# ============================================================
# 读取 Person test
# ============================================================

def collect_person_test():

    root = (
        os.path.join(os.environ.get("PERSON_DATA_ROOT", "data/person"), "test")
    )

    image_dir = os.path.join(
        root,
        "images"
    )

    label_dir = os.path.join(
        root,
        "labels"
    )

    files = []

    for name in sorted(
        os.listdir(image_dir)
    ):

        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            files.append(
                (
                    os.path.join(
                        image_dir,
                        name
                    ),
                    os.path.join(
                        label_dir,
                        name.rsplit(
                            ".",
                            1
                        )[0] + ".txt"
                    )
                )
            )

    return files


# ============================================================
# 读取 Apparatus test
# ============================================================

def collect_apparatus_test():

    root = (
        os.path.join(os.environ.get("APPARATUS_DATA_ROOT", "data/apparatus"), "test")
    )

    files = []

    for subset in [
        "ChemEq25",
        "SecondDataset"
    ]:

        image_dir = os.path.join(
            root,
            subset,
            "images"
        )

        label_dir = os.path.join(
            root,
            subset,
            "labels"
        )

        for name in sorted(
            os.listdir(image_dir)
        ):

            if name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                files.append(
                    (
                        os.path.join(
                            image_dir,
                            name
                        ),
                        os.path.join(
                            label_dir,
                            name.rsplit(
                                ".",
                                1
                            )[0] + ".txt"
                        )
                    )
                )

    return files


# ============================================================
# 模型推理并收集预测与 GT
# ============================================================

def collect_predictions(
    model,
    image_files,
    task
):

    predictions_by_image = []
    targets_by_image = []

    for image_id, (
        image_path,
        label_path
    ) in enumerate(
        tqdm(
            image_files,
            desc=f"Predict {task}"
        )
    ):

        image = cv2.imread(
            image_path
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        h, w = image.shape[:2]

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        resized = cv2.resize(
            rgb,
            (
                image_size,
                image_size
            )
        )

        tensor = torch.from_numpy(
            resized
        )

        tensor = (
            tensor
            .permute(2, 0, 1)
            .float()
            /
            255.0
        )

        tensor = (
            tensor
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            outputs = model(
                tensor
            )

        pred = outputs[task][0]

        detections = (
            non_max_suppression(
                pred,
                conf_thres=ap_conf_threshold,
                iou_thres=nms_iou_threshold
            )[0]
        )

        image_predictions = []

        if (
            detections is not None
            and
            len(detections) > 0
        ):

            for det in detections:

                (
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    cls_id
                ) = det.tolist()

                # 模型预测坐标来自 640x640
                # 映射回原图坐标
                x1 *= w / image_size
                x2 *= w / image_size

                y1 *= h / image_size
                y2 *= h / image_size

                image_predictions.append(
                    {
                        "image_id":
                            image_id,

                        "cls":
                            int(cls_id),

                        "conf":
                            float(conf),

                        "box": [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    }
                )

        targets = load_yolo_labels(
            label_path,
            w,
            h
        )

        predictions_by_image.append(
            image_predictions
        )

        targets_by_image.append(
            targets
        )

    return (
        predictions_by_image,
        targets_by_image
    )


# ============================================================
# 单类别 + 指定 IoU 的 AP
# ============================================================

def evaluate_class_ap(
    predictions_by_image,
    targets_by_image,
    class_id,
    iou_threshold
):

    predictions = []

    gt_by_image = {}

    total_gt = 0

    # 收集这个类别的 GT
    for image_id, targets in enumerate(
        targets_by_image
    ):

        class_targets = [
            target
            for target in targets
            if target["cls"] == class_id
        ]

        gt_by_image[image_id] = (
            class_targets
        )

        total_gt += len(
            class_targets
        )

    if total_gt == 0:
        return None

    # 收集这个类别的 prediction
    for image_predictions in (
        predictions_by_image
    ):

        for pred in image_predictions:

            if pred["cls"] == class_id:

                predictions.append(
                    pred
                )

    predictions.sort(
        key=lambda x: x["conf"],
        reverse=True
    )

    if len(predictions) == 0:
        return 0.0

    matched = {
        image_id:
            set()
        for image_id
        in range(
            len(targets_by_image)
        )
    }

    tp = np.zeros(
        len(predictions),
        dtype=np.float32
    )

    fp = np.zeros(
        len(predictions),
        dtype=np.float32
    )

    for pred_index, pred in enumerate(
        predictions
    ):

        image_id = pred[
            "image_id"
        ]

        targets = gt_by_image[
            image_id
        ]

        best_iou = 0.0
        best_gt_index = -1

        for gt_index, target in enumerate(
            targets
        ):

            if gt_index in matched[
                image_id
            ]:
                continue

            iou = box_iou(
                pred["box"],
                target["box"]
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = (
                    gt_index
                )

        if (
            best_gt_index >= 0
            and
            best_iou >= iou_threshold
        ):

            tp[pred_index] = 1.0

            matched[
                image_id
            ].add(
                best_gt_index
            )

        else:

            fp[pred_index] = 1.0

    cumulative_tp = np.cumsum(
        tp
    )

    cumulative_fp = np.cumsum(
        fp
    )

    recall = (
        cumulative_tp
        /
        (total_gt + 1e-16)
    )

    precision = (
        cumulative_tp
        /
        (
            cumulative_tp
            +
            cumulative_fp
            +
            1e-16
        )
    )

    return compute_ap(
        recall,
        precision
    )


# ============================================================
# Precision / Recall
# conf=0.25, IoU=0.50
# ============================================================

def compute_precision_recall(
    predictions_by_image,
    targets_by_image,
    class_ids
):

    tp = 0
    fp = 0
    fn = 0

    for (
        image_predictions,
        targets
    ) in zip(
        predictions_by_image,
        targets_by_image
    ):

        predictions = [
            pred
            for pred in image_predictions
            if (
                pred["conf"]
                >=
                pr_conf_threshold
                and
                pred["cls"]
                in class_ids
            )
        ]

        predictions.sort(
            key=lambda x:
                x["conf"],
            reverse=True
        )

        matched_gt = set()

        for pred in predictions:

            best_iou = 0.0
            best_index = -1

            for index, target in enumerate(
                targets
            ):

                if index in matched_gt:
                    continue

                if (
                    pred["cls"]
                    !=
                    target["cls"]
                ):
                    continue

                iou = box_iou(
                    pred["box"],
                    target["box"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best_index = index

            if (
                best_index >= 0
                and
                best_iou >= 0.50
            ):

                tp += 1

                matched_gt.add(
                    best_index
                )

            else:

                fp += 1

        relevant_targets = [
            target
            for target in targets
            if target["cls"]
            in class_ids
        ]

        fn += (
            len(relevant_targets)
            -
            len(matched_gt)
        )

    precision = (
        tp
        /
        (tp + fp + 1e-16)
    )

    recall = (
        tp
        /
        (tp + fn + 1e-16)
    )

    return (
        precision,
        recall,
        tp,
        fp,
        fn
    )


# ============================================================
# 一个任务的完整评估
# ============================================================

def evaluate_task(
    model,
    task,
    image_files,
    class_names
):

    print()
    print(
        "=" * 70
    )
    print(
        f"Evaluating: {task.upper()}"
    )
    print(
        "=" * 70
    )

    print(
        "Images:",
        len(image_files)
    )

    (
        predictions_by_image,
        targets_by_image
    ) = collect_predictions(
        model,
        image_files,
        task
    )

    class_ids = list(
        range(
            len(class_names)
        )
    )

    (
        precision,
        recall,
        tp,
        fp,
        fn
    ) = compute_precision_recall(
        predictions_by_image,
        targets_by_image,
        class_ids
    )

    ap_matrix = {}

    for class_id in class_ids:

        class_aps = []

        for iou_threshold in (
            iou_thresholds
        ):

            ap = evaluate_class_ap(
                predictions_by_image,
                targets_by_image,
                class_id,
                iou_threshold
            )

            class_aps.append(
                ap
            )

        ap_matrix[
            class_id
        ] = class_aps

    # 只平均测试集中真正存在GT的类别
    valid_classes = [
        class_id
        for class_id
        in class_ids
        if (
            ap_matrix[class_id][0]
            is not None
        )
    ]

    if not valid_classes:

        map50 = 0.0
        map5095 = 0.0

    else:

        map50 = np.mean(
            [
                ap_matrix[
                    class_id
                ][0]
                for class_id
                in valid_classes
            ]
        )

        map5095 = np.mean(
            [
                np.mean(
                    [
                        ap
                        for ap
                        in ap_matrix[
                            class_id
                        ]
                        if ap is not None
                    ]
                )
                for class_id
                in valid_classes
            ]
        )

    print()
    print(
        "Per-class AP:"
    )

    print(
        f"{'Class':45s}"
        f"{'AP50':>10s}"
        f"{'AP50-95':>12s}"
    )

    print(
        "-" * 67
    )

    for class_id in valid_classes:

        class_ap50 = (
            ap_matrix[
                class_id
            ][0]
        )

        class_map = np.mean(
            [
                ap
                for ap
                in ap_matrix[
                    class_id
                ]
                if ap is not None
            ]
        )

        print(
            f"{class_names[class_id][:44]:45s}"
            f"{class_ap50:10.4f}"
            f"{class_map:12.4f}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        f"{task.upper()} FINAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "TP:",
        tp
    )

    print(
        "FP:",
        fp
    )

    print(
        "FN:",
        fn
    )

    print(
        "Precision @ conf=0.25, IoU=0.50:",
        round(
            precision,
            4
        )
    )

    print(
        "Recall    @ conf=0.25, IoU=0.50:",
        round(
            recall,
            4
        )
    )

    print(
        "mAP50:",
        round(
            float(map50),
            4
        )
    )

    print(
        "mAP50-95:",
        round(
            float(map5095),
            4
        )
    )

    return {
        "precision":
            precision,

        "recall":
            recall,

        "map50":
            map50,

        "map5095":
            map5095
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Device:",
        device
    )

    print(
        "Loading model:",
        model_path
    )

    model = DualHeadYOLO()

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    print(
        "Model loaded!"
    )

    person_files = (
        collect_person_test()
    )

    apparatus_files = (
        collect_apparatus_test()
    )

    person_result = evaluate_task(
        model=model,
        task="person",
        image_files=person_files,
        class_names=person_names
    )

    apparatus_result = evaluate_task(
        model=model,
        task="apparatus",
        image_files=apparatus_files,
        class_names=apparatus_names
    )

    print()
    print(
        "=" * 70
    )
    print(
        "DUAL HEAD SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        f"{'Task':15s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'mAP50':>12s}"
        f"{'mAP50-95':>14s}"
    )

    print(
        "-" * 65
    )

    print(
        f"{'Person':15s}"
        f"{person_result['precision']:12.4f}"
        f"{person_result['recall']:12.4f}"
        f"{person_result['map50']:12.4f}"
        f"{person_result['map5095']:14.4f}"
    )

    print(
        f"{'Apparatus':15s}"
        f"{apparatus_result['precision']:12.4f}"
        f"{apparatus_result['recall']:12.4f}"
        f"{apparatus_result['map50']:12.4f}"
        f"{apparatus_result['map5095']:14.4f}"
    )


if __name__ == "__main__":

    main()