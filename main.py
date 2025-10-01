import torch
from ssh import SSH, train_epoch, validate
from backbone import MobileNetV2Backbone
from cfg import device
from utils import (
    generate_anchors,
)
from dataset import train_loader, test_loader, val_loader
from ssh import test_and_visualize


def train():
    model = SSH(MobileNetV2Backbone()).to(device)

    anchors_list = [
        generate_anchors((40, 40), stride=16, scales=[16, 32]),
        generate_anchors((20, 20), stride=32, scales=[64, 128]),
        generate_anchors((10, 10), stride=64, scales=[256, 512]),
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 50}")

        # Training
        train_loss, train_cls, train_reg = train_epoch(
            model, train_loader, optimizer, anchors_list, epoch + 1
        )
        print(
            f"Train: Loss: {train_loss:.4f} | Cls: {train_cls:.4f} | Reg: {train_reg:.4f}"
        )

        # Validation
        val_loss, val_cls, val_reg = validate(
            model, val_loader, anchors_list, epoch + 1
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✓ Saved best model with validation loss: {val_loss:.4f}")

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
    test_and_visualize(model, test_loader, anchors_list)


def main():
    anchors_list = [
        generate_anchors((40, 40), stride=16, scales=[16, 32]),
        generate_anchors((20, 20), stride=32, scales=[64, 128]),
        generate_anchors((10, 10), stride=64, scales=[256, 512]),
    ]
    checkpoint = torch.load("./best_model.pth", map_location=torch.device("cpu"))
    model = SSH(MobileNetV2Backbone()).to(device)
    model.load_state_dict(checkpoint)
    test_and_visualize(model, val_loader, anchors_list, n_out=5)


if __name__ == "__main__":
    main()
