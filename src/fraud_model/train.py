from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_model.config import ArtifactPaths
from fraud_model.data import infer_feature_columns, load_csv, load_ieee_cis
from fraud_model.features import FeaturePreprocessor
from fraud_model.model import TransactionTransformer


def build_loader(numeric, categorical, labels, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(numeric, dtype=torch.float32),
        torch.tensor(categorical, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(args: argparse.Namespace) -> dict[str, float]:
    data = load_ieee_cis(args.ieee_cis_dir) if args.ieee_cis_dir else load_csv(args.data)
    if args.sample_rows and args.sample_rows < len(data):
        data = data.sample(n=args.sample_rows, random_state=args.seed)

    numeric_features, categorical_features = infer_feature_columns(data, args.label_column)
    train_frame, valid_frame = train_test_split(
        data,
        test_size=args.validation_size,
        stratify=data[args.label_column],
        random_state=args.seed,
    )

    preprocessor = FeaturePreprocessor.fit(train_frame, numeric_features, categorical_features)
    train_numeric, train_categorical = preprocessor.transform(train_frame)
    valid_numeric, valid_categorical = preprocessor.transform(valid_frame)
    train_labels = train_frame[args.label_column].to_numpy()
    valid_labels = valid_frame[args.label_column].to_numpy()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = TransactionTransformer(
        num_numeric_features=train_numeric.shape[1],
        category_cardinalities=preprocessor.category_cardinalities(),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience
    )
    positive_count = float(train_labels.sum())
    negative_count = float(len(train_labels) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = build_loader(train_numeric, train_categorical, train_labels, args.batch_size, shuffle=True)

    history = []
    best_average_precision = -1.0
    best_state = None
    best_epoch = 0
    best_probabilities = None
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for numeric, categorical, labels in train_loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(numeric, categorical), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(valid_numeric, dtype=torch.float32, device=device),
                torch.tensor(valid_categorical, dtype=torch.long, device=device),
            )
            epoch_probabilities = torch.sigmoid(logits).cpu().numpy()
        epoch_average_precision = float(average_precision_score(valid_labels, epoch_probabilities))
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": sum(losses) / max(len(losses), 1),
            "roc_auc": float(roc_auc_score(valid_labels, epoch_probabilities))
            if len(set(valid_labels)) > 1
            else 0.0,
            "average_precision": epoch_average_precision,
        }
        history.append(epoch_metrics)
        scheduler.step(epoch_average_precision)
        if epoch_average_precision > best_average_precision:
            best_average_precision = epoch_average_precision
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_probabilities = epoch_probabilities
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    probabilities = best_probabilities
    if probabilities is None:
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(valid_numeric, dtype=torch.float32, device=device),
                torch.tensor(valid_categorical, dtype=torch.long, device=device),
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= args.threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(valid_labels, probabilities)) if len(set(valid_labels)) > 1 else 0.0,
        "average_precision": float(average_precision_score(valid_labels, probabilities)),
        "f1": float(f1_score(valid_labels, predictions, zero_division=0)),
        "threshold": float(args.threshold),
        "device": str(device),
        "rows": int(len(data)),
        "best_epoch": int(best_epoch),
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "history": history,
    }

    artifacts = ArtifactPaths(Path(args.artifact_dir))
    artifacts.artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(artifacts.preprocessor)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_numeric_features": train_numeric.shape[1],
            "category_cardinalities": preprocessor.category_cardinalities(),
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "best_epoch": best_epoch,
        },
        artifacts.model,
    )
    artifacts.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transaction fraud Transformer.")
    parser.add_argument("--data", help="Path to a CSV file of transactions.")
    parser.add_argument("--ieee-cis-dir", help="Folder containing train_transaction.csv and train_identity.csv.")
    parser.add_argument("--label-column", default="isFraud")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--sample-rows", type=int, help="Optional row sample for a quick Kaggle sanity run.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override automatic device selection.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))
