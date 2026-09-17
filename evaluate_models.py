import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json


def get_baseline_model(num_classes):
    class BaselineCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)
    return BaselineCNN(num_classes)


def get_transfer_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def evaluate(model, loader, device, class_names, model_name):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names)

    print(f"\n{'='*60}")
    print(f"Results: {model_name}")
    print('='*60)
    print(report_text)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    plt.savefig(f'confusion_matrix_{safe_name}.png')
    plt.close()

    return report, report_text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = "data/tomato_split"

    # --- Test transforms (matching each model's training resolution) ---
    transform_128 = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset_128 = datasets.ImageFolder(f"{data_dir}/test", transform=transform_128)
    test_dataset_224 = datasets.ImageFolder(f"{data_dir}/test", transform=transform_224)

    class_names = test_dataset_128.classes
    num_classes = len(class_names)

    test_loader_128 = DataLoader(test_dataset_128, batch_size=32, shuffle=False, num_workers=2)
    test_loader_224 = DataLoader(test_dataset_224, batch_size=32, shuffle=False, num_workers=2)

    all_results = {}

    # --- 1. Baseline CNN ---
    baseline = get_baseline_model(num_classes)
    baseline.load_state_dict(torch.load("baseline_cnn_best.pth", map_location=device))
    baseline = baseline.to(device)
    report, text = evaluate(baseline, test_loader_128, device, class_names, "Baseline CNN")
    all_results["baseline_cnn"] = {"accuracy": report["accuracy"], "report": report}

    # --- 2. Frozen Transfer Learning ---
    frozen = get_transfer_model(num_classes)
    frozen.load_state_dict(torch.load("transfer_mobilenet_best.pth", map_location=device))
    frozen = frozen.to(device)
    report, text = evaluate(frozen, test_loader_224, device, class_names, "Frozen Transfer Learning")
    all_results["frozen_transfer"] = {"accuracy": report["accuracy"], "report": report}

    # --- 3. Fine-tuned Transfer Learning ---
    finetuned = get_transfer_model(num_classes)
    finetuned.load_state_dict(torch.load("finetuned_mobilenet_best.pth", map_location=device))
    finetuned = finetuned.to(device)
    report, text = evaluate(finetuned, test_loader_224, device, class_names, "Fine-tuned Transfer Learning")
    all_results["finetuned_transfer"] = {"accuracy": report["accuracy"], "report": report}

    # --- Summary ---
    print(f"\n{'='*60}")
    print("FINAL TEST SET SUMMARY")
    print('='*60)
    for name, res in all_results.items():
        print(f"{name}: {res['accuracy']*100:.2f}%")

    with open("final_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll confusion matrices and results saved!")


if __name__ == "__main__":
    main()