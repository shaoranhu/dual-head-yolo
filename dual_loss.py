import torch

from ultralytics.utils.loss import v8DetectionLoss

from loss_wrapper import LossModelWrapper



class DualLoss:

    def __init__(self, model, person_weight=1.0, apparatus_weight=1.0):

        # Task weights are independent of box/cls/dfl component gains.
        self.person_weight = float(person_weight)
        self.apparatus_weight = float(apparatus_weight)
        if not (0 < self.person_weight < float('inf')):
            raise ValueError('person_weight must be finite and positive')
        if not (0 < self.apparatus_weight < float('inf')):
            raise ValueError('apparatus_weight must be finite and positive')


        # =========================
        # person detection loss
        # =========================

        self.person_wrapper = LossModelWrapper(
            model.person_head,
            model.args
        )


        self.person_loss = v8DetectionLoss(
            self.person_wrapper
        )



        # =========================
        # apparatus detection loss
        # =========================

        self.apparatus_wrapper = LossModelWrapper(
            model.apparatus_head,
            model.args
        )


        self.apparatus_loss = v8DetectionLoss(
            self.apparatus_wrapper
        )



    def __call__(
        self,
        outputs,
        batch
    ):


        total_loss = 0



        # -------------------------
        # person loss
        # -------------------------

        if "person" in batch:


            loss_person, _ = self.person_loss(
                outputs["person"],
                batch["person"]
            )


            total_loss += self.person_weight * loss_person



        # -------------------------
        # apparatus loss
        # -------------------------

        if "apparatus" in batch:


            loss_apparatus, _ = self.apparatus_loss(
                outputs["apparatus"],
                batch["apparatus"]
            )


            total_loss += self.apparatus_weight * loss_apparatus



        return total_loss