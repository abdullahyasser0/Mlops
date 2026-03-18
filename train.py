import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch


# ── Model ────────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# ── Training helpers ─────────────────────────────────────────────────────────


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def build_eval_table(model, loader, device, max_rows=500):
    """Collect predictions on the test set; return a DataFrame for log_table.
    """
    model.eval()
    actuals, preds, confs = [], [], []
    seen = 0
    with torch.no_grad():
        for images, labels in loader:
            if seen >= max_rows:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            top_conf, top_pred = probs.max(dim=1)
            actuals.extend(labels.cpu().tolist())
            preds.extend(top_pred.cpu().tolist())
            confs.extend(top_conf.cpu().tolist())
            seen += labels.size(0)
    return pd.DataFrame({
        "actual": actuals[:max_rows],
        "predicted": preds[:max_rows],
        "confidence": [round(c, 4) for c in confs[:max_rows]],
        "correct": [
            int(a == p)
            for a, p in zip(actuals[:max_rows], preds[:max_rows])
        ],
    })


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9
    )

    mlflow.set_experiment("Assignment3_Abdullah_Yasser")

    with mlflow.start_run(run_name=args.run_name):
        # ── Tags ─────────────────────────────────────────────────────────────
        mlflow.set_tag("student_id", "202201083")
        mlflow.set_tag("student_name", "Abdullah Yasser")
        mlflow.set_tag("model", "MLP")
        mlflow.set_tag("dataset", "MNIST")

        # ── Parameters ───────────────────────────────────────────────────────
        mlflow.log_params({
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": "SGD",
            "momentum": 0.9,
            "hidden_units": "256-128",
        })

        # ── Training loop ────────────────────────────────────────────────────
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = eval_epoch(
                model, test_loader, criterion, device
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            print(
                f"Epoch {epoch:02d}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

        # ── Evaluation table ─────────────────────────────────────────────────
        eval_df = build_eval_table(model, test_loader, device)
        mlflow.log_table(
            data=eval_df,
            artifact_file="eval_results/test_evaluation.json"
        )

        # ── Save model ───────────────────────────────────────────────────────
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            mlflow.pytorch.save_model(model, tmp_dir)
            mlflow.log_artifacts(tmp_dir, artifact_path="mnist_mlp")
        print("Run finished. Model saved to MLflow.")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST MLP with MLflow tracking"
    )
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--run_name", type=str, default="default_run")
    args = parser.parse_args()
    main(args)
