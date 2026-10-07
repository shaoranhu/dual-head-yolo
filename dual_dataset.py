import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset



# =====================================================
# Image loader
# =====================================================

def load_image(path):

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (640,640)
    )

    image = np.array(image)


    image = (
        torch.from_numpy(image)
        .permute(2,0,1)
        .float()
        /255.0
    )

    return image



# =====================================================
# YOLO label loader
# =====================================================

def load_labels(path):

    classes = []
    boxes = []


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line=line.strip()

            if not line:
                continue


            values=line.split()


            if len(values)!=5:

                raise ValueError(
                    f"Label error: {path}\n{line}"
                )


            cls=float(values[0])


            bbox=[
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4])
            ]


            classes.append([cls])

            boxes.append(bbox)



    if len(classes)==0:

        cls=torch.zeros(
            (0,1),
            dtype=torch.float32
        )

        bboxes=torch.zeros(
            (0,4),
            dtype=torch.float32
        )


    else:

        cls=torch.tensor(
            classes,
            dtype=torch.float32
        )


        bboxes=torch.tensor(
            boxes,
            dtype=torch.float32
        )


    return cls,bboxes




# =====================================================
# Person Dataset
# =====================================================

class PersonDataset(Dataset):


    def __init__(self, root):


        self.image_dir=os.path.join(
            root,
            "images"
        )


        self.label_dir=os.path.join(
            root,
            "labels"
        )


        self.images=sorted([

            f for f in os.listdir(
                self.image_dir
            )

            if f.lower().endswith(
                (
                    ".jpg",
                    ".png",
                    ".jpeg"
                )
            )

        ])



    def __len__(self):

        return len(self.images)



    def __getitem__(self,idx):


        name=self.images[idx]


        image=load_image(

            os.path.join(
                self.image_dir,
                name
            )

        )


        label_path=os.path.join(

            self.label_dir,

            name.rsplit(".",1)[0]+".txt"

        )


        cls,bboxes=load_labels(
            label_path
        )


        return {

            "image":image,

            "cls":cls,

            "bboxes":bboxes,

            "task":"person"

        }




# =====================================================
# Apparatus Dataset
# =====================================================

class ApparatusDataset(Dataset):


    def __init__(
        self,
        root,
        split="train"
    ):


        self.image_dir=os.path.join(

            root,

            split,

            "images"

        )


        self.label_dir=os.path.join(

            root,

            split,

            "labels"

        )



        self.images=sorted([

            f for f in os.listdir(
                self.image_dir
            )

            if f.lower().endswith(
                (
                    ".jpg",
                    ".png",
                    ".jpeg"
                )
            )

        ])




    def __len__(self):

        return len(self.images)




    def __getitem__(self,idx):


        name=self.images[idx]


        image=load_image(

            os.path.join(
                self.image_dir,
                name
            )

        )


        label_path=os.path.join(

            self.label_dir,

            name.rsplit(".",1)[0]+".txt"

        )


        cls,bboxes=load_labels(
            label_path
        )



        return {

            "image":image,

            "cls":cls,

            "bboxes":bboxes,

            "task":"apparatus"

        }# =====================================================
# Dual Train Dataset
# =====================================================

class DualDataset(Dataset):


    def __init__(self):


        # Person train

        self.person = PersonDataset(

            os.path.join(os.environ.get("PERSON_DATA_ROOT", "data/person"), "train")

        )


        # Apparatus train

        self.apparatus = ApparatusDataset(

            os.environ.get("APPARATUS_DATA_ROOT", "data/apparatus"),

            split="train"

        )


        self.person_len = len(
            self.person
        )


        self.apparatus_len = len(
            self.apparatus
        )


        self.total = (

            self.person_len

            +

            self.apparatus_len

        )



    def __len__(self):

        return self.total



    def __getitem__(self,idx):


        if idx < self.person_len:

            return self.person[idx]


        else:

            return self.apparatus[

                idx-self.person_len

            ]




# =====================================================
# Dual Validation Dataset
# =====================================================

class DualValDataset(Dataset):


    def __init__(self):


        # Person validation

        self.person = PersonDataset(

            os.path.join(os.environ.get("PERSON_DATA_ROOT", "data/person"), "val")

        )


        # Apparatus validation

        self.apparatus = ApparatusDataset(

            os.environ.get("APPARATUS_DATA_ROOT", "data/apparatus"),

            split="valid"

        )


        self.person_len = len(
            self.person
        )


        self.apparatus_len = len(
            self.apparatus
        )


        self.total = (

            self.person_len

            +

            self.apparatus_len

        )



    def __len__(self):

        return self.total



    def __getitem__(self,idx):


        if idx < self.person_len:

            return self.person[idx]


        else:

            return self.apparatus[

                idx-self.person_len

            ]




# =====================================================
# Test
# =====================================================

if __name__ == "__main__":


    print(
        "========== TRAIN =========="
    )


    train_dataset = DualDataset()


    print(
        "Total:",
        len(train_dataset)
    )


    print(
        "Person:",
        len(train_dataset.person)
    )


    print(
        "Apparatus:",
        len(train_dataset.apparatus)
    )



    print(
        "\n========== VALIDATION =========="
    )


    val_dataset = DualValDataset()


    print(
        "Total:",
        len(val_dataset)
    )


    print(
        "Person:",
        len(val_dataset.person)
    )


    print(
        "Apparatus:",
        len(val_dataset.apparatus)
    )



    print(
        "\n========== SAMPLE =========="
    )


    print(
        "Train first:",
        train_dataset[0]["task"]
    )


    print(
        "Train apparatus:",
        train_dataset[
            train_dataset.person_len
        ]["task"]
    )


    print(
        "Val first:",
        val_dataset[0]["task"]
    )


    print(
        "Val apparatus:",
        val_dataset[
            val_dataset.person_len
        ]["task"]
    )