import logging

logging.disable(logging.CRITICAL)


from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import torch
from ssh import SSH, train_epoch, validate
from backbone import MobileNetV2Backbone
from cfg import device
from utils import generate_anchors, draw_bbox_and_save
from dataset import train_loader, test_loader, val_loader
from ssh import test_and_visualize
from PIL import Image


def train():
    model = SSH(MobileNetV2Backbone()).to(device)

    anchors_list = [
        generate_anchors((160, 160), stride=4, scales=[8, 16]),
        generate_anchors((80, 80), stride=8, scales=[16, 32]),
        generate_anchors((40, 40), stride=16, scales=[32, 64]),
    ]

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    # Gradient clipping value
    max_grad_norm = 1.0

    num_epochs = 30  # Increased number of epochs
    best_val_loss = float("inf")

    # Early stopping
    patience = 5
    epochs_no_improve = 0
    min_val_loss = float("inf")

    # Initialize tqdm progress bar for epochs
    epoch_pbar = tqdm(
        range(num_epochs), desc="Training Progress", position=0, leave=True
    )

    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        # Training
        train_loss, train_cls, train_reg = train_epoch(
            model, train_loader, optimizer, anchors_list, epoch + 1, max_grad_norm
        )

        # Validation
        with torch.no_grad():
            val_loss, val_cls, val_reg = validate(
                model, val_loader, anchors_list, epoch + 1
            )

        # Step the scheduler on validation loss
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        # Update progress bar with current metrics
        epoch_pbar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}",
                "Best Val": f"{best_val_loss:.4f}",
            }
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            epoch_pbar.write(
                f"✓ New best model saved with validation loss: {val_loss:.4f}"
            )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            f"checkpoint_epoch_{epoch + 1}.pth",
        )

    # Test and visualize after training
    print("\n" + "=" * 50)
    print("Generating test visualizations...")
    print("=" * 50)
    test_and_visualize(model, test_loader, anchors_list, n_out=5)


def main():
    # anchors_list = [
    #     generate_anchors((160, 160), stride=4, scales=[8, 16]),
    #     generate_anchors((80, 80), stride=8, scales=[16, 32]),
    #     generate_anchors((40, 40), stride=16, scales=[32, 64]),
    # ]
    # checkpoint = torch.load(
    #     "./checkpoint_epoch_5.pth", map_location=torch.device("cpu")
    # )
    # model = SSH(MobileNetV2Backbone()).to(device)
    # # model.load_state_dict(checkpoint)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # test_and_visualize(model, train_loader, anchors_list, n_out=5)

    train()


if __name__ == "__main__":
    main()
