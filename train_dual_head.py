import os
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from dual_dataset import (
    DualDataset,
    DualValDataset
)

from dual_head_model import DualHeadYOLO

from dual_collate import dual_collate

from dual_loss import DualLoss



# =========================
# Training settings
# =========================

epochs = 5
learning_rate = 1e-5
finetune_weights = os.environ.get("DUAL_FINETUNE_WEIGHTS", "checkpoints/dual_head_best.pt")

batch_size = 1

# Experimental task weighting; keep batches single-task.
person_loss_weight = 4.0
apparatus_loss_weight = 1.0
experiment_tag = f'finetune_p{person_loss_weight:g}_a{apparatus_loss_weight:g}'


checkpoint_path = (
    f"dual_head_{experiment_tag}_checkpoint.pt"
)


best_model_path = (
    f"dual_head_{experiment_tag}_best.pt"
)



device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# =========================
# Save checkpoint
# =========================

def save_checkpoint(
        model,
        optimizer,
        epoch,
        best_val_loss
):

    checkpoint = {


        "epoch":
            epoch,


        "best_val_loss":
            best_val_loss,


        "model_state_dict":
            model.state_dict(),


        "optimizer_state_dict":
            optimizer.state_dict()

    }



    torch.save(
        checkpoint,
        checkpoint_path
    )



# =========================
# Validation
# =========================

def validate(
        model,
        val_loader,
        criterion
):

    model.eval()


    total_loss = 0



    with torch.no_grad():


        progress = tqdm(

            val_loader,

            desc="Validation"

        )


        for batch in progress:


            images = batch["image"].to(

                device,

                non_blocking=True

            )


            outputs = model(

                images

            )


            target = {}



            for key,value in batch.items():


                if key == "image":

                    continue



                target[key] = {


                    k:

                    v.to(device)


                    for k,v in value.items()

                }



            loss = criterion(

                outputs,

                target

            )



            if loss.ndim > 0:

                loss = loss.sum()



            total_loss += loss.item()



            progress.set_postfix(

                loss=loss.item()

            )



    model.train()


    return total_loss# =========================
# Main
# =========================

def main():


    # =====================
    # Dataset
    # =====================


    train_dataset = DualDataset()


    val_dataset = DualValDataset()



    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=4,

        pin_memory=True,

        collate_fn=dual_collate

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=4,

        pin_memory=True,

        collate_fn=dual_collate

    )



    print(
        "Train size:",
        len(train_dataset)
    )


    print(
        "Validation size:",
        len(val_dataset)
    )



    # =====================
    # Model
    # =====================


    model = DualHeadYOLO()
    model.load_state_dict(torch.load(finetune_weights, map_location='cpu', weights_only=True))
    # Ensure the shared pretrained network participates in fine-tuning.
    model.shared_layers.requires_grad_(True)
    print(f'Fine-tuning from {finetune_weights}; epochs={epochs}, lr={learning_rate}')


    model.to(device)


    model.train()



    # =====================
    # Loss
    # =====================


    criterion = DualLoss(
        model,
        person_weight=person_loss_weight,
        apparatus_weight=apparatus_loss_weight,
    )
    print(f'Task loss weights: person={person_loss_weight}, apparatus={apparatus_loss_weight}')



    # =====================
    # Optimizer
    # =====================


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=learning_rate

    )



    # =====================
    # Resume checkpoint
    # =====================


    start_epoch = 0


    best_val_loss = float("inf")



    try:


        checkpoint = torch.load(

            checkpoint_path,

            map_location=device

        )


        model.load_state_dict(

            checkpoint["model_state_dict"]

        )


        optimizer.load_state_dict(

            checkpoint["optimizer_state_dict"]

        )


        start_epoch = checkpoint["epoch"]



        best_val_loss = checkpoint.get(

            "best_val_loss",

            float("inf")

        )



        print(

            "Resume from epoch:",

            start_epoch

        )


        print(

            "Best val loss:",

            best_val_loss

        )



    except FileNotFoundError:


        print(

            "No fine-tuning checkpoint found; starting from the previous best weights."

        )    # =====================
    # Training loop
    # =====================


    for epoch in range(

        start_epoch,

        epochs

    ):



        model.train()


        total_train_loss = 0



        progress = tqdm(

            train_loader,

            desc=f"Epoch {epoch+1}/{epochs}"

        )



        for batch in progress:



            images = batch["image"].to(

                device,

                non_blocking=True

            )


            outputs = model(

                images

            )



            target = {}



            for key,value in batch.items():


                if key == "image":

                    continue



                target[key] = {


                    k:

                    v.to(device)


                    for k,v in value.items()

                }



            loss = criterion(

                outputs,

                target

            )



            if loss.ndim > 0:

                loss = loss.sum()



            optimizer.zero_grad()


            loss.backward()


            optimizer.step()



            total_train_loss += loss.item()



            progress.set_postfix(

                loss=loss.item()

            )



        # =====================
        # Validation
        # =====================


        val_loss = validate(

            model,

            val_loader,

            criterion

        )



        print()


        print(

            f"Epoch {epoch+1}/{epochs}"

        )


        print(

            "Train loss:",

            total_train_loss

        )


        print(

            "Validation loss:",

            val_loss

        )



        # =====================
        # Save checkpoint
        # =====================


        save_checkpoint(

            model,

            optimizer,

            epoch + 1,

            best_val_loss

        )


        print(

            "Checkpoint saved!"

        )



        # =====================
        # Save best model
        # =====================


        if val_loss < best_val_loss:


            best_val_loss = val_loss



            torch.save(

                model.state_dict(),

                best_model_path

            )



            print(

                "Best model saved!"

            )



            # 更新checkpoint中的best值

            save_checkpoint(

                model,

                optimizer,

                epoch + 1,

                best_val_loss

            )



    # =====================
    # Save final model
    # =====================


    torch.save(

        model.state_dict(),

        f"dual_head_{experiment_tag}_final.pt"

    )



    print(

        "Training finished!"

    )


    print(

        f"Saved: dual_head_{experiment_tag}_final.pt"

    )





if __name__ == "__main__":

    main()