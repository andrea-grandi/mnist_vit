import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ViT


def train_mnist_vit(save_path="vit_mnist_model.pt"):
    batch_size = 64
    epochs = 1
    learning_rate = 0.001

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Usando dispositivo MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Usando dispositivo CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Usando dispositivo CPU")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ViT(
        image_size=32,
        patch_size=4,
        in_channels=1,
        hidden_size=256,
        num_classes=10,
        num_layers=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch: {epoch + 1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                    f"Loss: {running_loss / (batch_idx + 1):.4f}, "
                    f"Acc: {100.0 * correct / total:.2f}%"
                )

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                # Sposta i dati sul dispositivo appropriato
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100.0 * correct / total
        print(
            f"\nTest set: Average loss: {test_loss / len(test_loader):.4f}, "
            f"Accuracy: {accuracy:.2f}%\n"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            os.makedirs(
                os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": test_loss / len(test_loader),
                    "accuracy": accuracy,
                },
                save_path,
            )

            print(
                f"Miglior modello salvato con accuratezza: {accuracy:.2f}% in: {save_path}"
            )

    last_model_path = (
        f"{os.path.splitext(save_path)[0]}_last{os.path.splitext(save_path)[1]}"
    )
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": test_loss / len(test_loader),
            "accuracy": accuracy,
        },
        last_model_path,
    )
    print(f"Ultimo modello salvato in: {last_model_path}")

    return model, best_accuracy


def load_model(model_path, model_class=None, device=None):
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    checkpoint = torch.load(model_path, map_location=device)

    if model_class is not None:
        model = model_class.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint
    else:
        return checkpoint


def visualize_predictions(model, test_loader, device, num_images=5):
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    images = images[:num_images].to(device)
    labels = labels[:num_images]

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i in range(num_images):
        axes[i].imshow(images[i].cpu().squeeze(), cmap="gray")
        axes[i].set_title(f"Pred: {predicted[i].item()}, True: {labels[i].item()}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Usando dispositivo MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Usando dispositivo CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Usando dispositivo CPU")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    save_path = "models/vit_mnist_best.pt"

    model, best_accuracy = train_mnist_vit(save_path=save_path)
    print(f"Addestramento completato. Miglior accuratezza: {best_accuracy:.2f}%")

    visualize_predictions(model, test_loader, device)
