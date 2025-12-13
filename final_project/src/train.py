"""
Training script for sign language recognition models.

Usage:
    python src/train.py --model lstm --config configs/config.yaml
    python src/train.py --model transformer --config configs/config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm

from dataset import get_data_loaders
from models.lstm import SignLSTM, SignLSTMWithAttention
from models.transformer import SignTransformer, SignTransformerSimple


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / len(train_loader), correct / total


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / len(data_loader), correct / total


def create_model(model_type: str, num_classes: int, config: dict) -> nn.Module:
    """Create model based on type."""
    model_config = config.get('model', {})

    if model_type == 'lstm':
        return SignLSTM(
            input_dim=model_config.get('input_dim', 63),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 2),
            num_classes=num_classes,
            dropout=model_config.get('dropout', 0.3)
        )
    elif model_type == 'lstm_attention':
        return SignLSTMWithAttention(
            input_dim=model_config.get('input_dim', 63),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 2),
            num_classes=num_classes,
            dropout=model_config.get('dropout', 0.3)
        )
    elif model_type == 'transformer':
        return SignTransformer(
            input_dim=model_config.get('input_dim', 63),
            d_model=model_config.get('hidden_dim', 256),
            num_classes=num_classes,
            dropout=model_config.get('dropout', 0.1)
        )
    elif model_type == 'transformer_simple':
        return SignTransformerSimple(
            input_dim=model_config.get('input_dim', 63),
            d_model=model_config.get('hidden_dim', 128),
            num_classes=num_classes,
            dropout=model_config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Train sign language recognition model")
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["lstm", "lstm_attention", "transformer", "transformer_simple"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    data_config = config.get('data', {})
    train_config = config.get('training', {})

    train_loader, val_loader, test_loader = get_data_loaders(
        landmarks_dir=data_config.get('landmarks_path', 'data/landmarks'),
        annotations_file=f"{data_config.get('dataset_path', 'data/wlasl')}/WLASL_v0.3.json",
        batch_size=train_config.get('batch_size', 32),
        max_seq_length=data_config.get('max_seq_length', 60)
    )

    num_classes = train_loader.dataset.dataset.num_classes
    print(f"Number of classes: {num_classes}")

    # Create model
    model = create_model(args.model, num_classes, config)
    model = model.to(device)
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=train_config.get('learning_rate', 0.001))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    epochs = train_config.get('epochs', 50)
    best_val_acc = 0
    patience_counter = 0
    early_stopping_patience = train_config.get('early_stopping_patience', 10)

    checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_dir / f"best_{args.model}.pt")
            print(f"Saved best model with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    checkpoint = torch.load(checkpoint_dir / f"best_{args.model}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
