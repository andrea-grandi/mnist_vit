import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ViT


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
    images_cpu = images.cpu()

    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for i in range(num_images):
        axes[i].imshow(images_cpu[i].squeeze(), cmap="gray")
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

    model_class = lambda: ViT(
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
    model, checkpoint = load_model(save_path, model_class(), device)
    print(
        f"Modello caricato. Epoca: {checkpoint['epoch']}, Accuratezza: {checkpoint['accuracy']:.2f}%"
    )

    visualize_predictions(model, test_loader, device)
