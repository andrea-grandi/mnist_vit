import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ViT


def train_mnist_vit():
    batch_size = 64
    epochs = 1
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        print(
            f"\nTest set: Average loss: {test_loss / len(test_loader):.4f}, "
            f"Accuracy: {100.0 * correct / total:.2f}%\n"
        )

    return model


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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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

    model = train_mnist_vit()

    visualize_predictions(model, test_loader, device)
