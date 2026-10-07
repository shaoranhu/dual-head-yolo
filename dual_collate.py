import torch



def dual_collate(batch):
    """
    将混合任务 batch 拆成人员任务和器皿任务

    输入:
    [
      {
        image,
        cls,
        bboxes,
        task
      },
      ...
    ]

    输出:

    {
      image: 所有图片,

      person:
          {
             batch_idx,
             cls,
             bboxes
          },

      apparatus:
          {
             batch_idx,
             cls,
             bboxes
          }
    }

    """

    images = torch.stack(
        [
            item["image"]
            for item in batch
        ]
    )


    person_cls = []
    person_boxes = []
    person_idx = []


    apparatus_cls = []
    apparatus_boxes = []
    apparatus_idx = []



    for i, item in enumerate(batch):


        if item["task"] == "person":


            person_cls.append(
                item["cls"]
            )


            person_boxes.append(
                item["bboxes"]
            )


            person_idx.append(
                torch.full(
                    (len(item["cls"]),),
                    i
                )
            )



        elif item["task"] == "apparatus":


            apparatus_cls.append(
                item["cls"]
            )


            apparatus_boxes.append(
                item["bboxes"]
            )


            apparatus_idx.append(
                torch.full(
                    (len(item["cls"]),),
                    i
                )
            )



    output = {

        "image": images

    }



    # -------------------------
    # person
    # -------------------------

    if len(person_cls) > 0:


        output["person"] = {

            "cls":
                torch.cat(
                    person_cls,
                    dim=0
                ),

            "bboxes":
                torch.cat(
                    person_boxes,
                    dim=0
                ),

            "batch_idx":
                torch.cat(
                    person_idx,
                    dim=0
                )

        }



    # -------------------------
    # apparatus
    # -------------------------

    if len(apparatus_cls) > 0:


        output["apparatus"] = {

            "cls":
                torch.cat(
                    apparatus_cls,
                    dim=0
                ),

            "bboxes":
                torch.cat(
                    apparatus_boxes,
                    dim=0
                ),

            "batch_idx":
                torch.cat(
                    apparatus_idx,
                    dim=0
                )

        }



    return output