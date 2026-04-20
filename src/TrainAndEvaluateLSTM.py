# divya bharathi

from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

Article = Tuple[str, int]


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _build_vocab(texts: Sequence[str], max_words: int = 20000) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(_tokenize(text))

    vocab = {word: i + 2 for i, (word, _) in enumerate(counter.most_common(max_words))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


def _encode(text: str, vocab: Dict[str, int], max_len: int = 300) -> List[int]:
    ids = [vocab.get(tok, 1) for tok in _tokenize(text)]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


def _read_test_articles() -> List[Article]:
    all_test_articles: List[Article] = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    real_test_path = os.path.join(project_root, "Data", "test", "RealNewsArticlesTestingSet")
    fake_test_path = os.path.join(project_root, "Data", "test", "FakeNewsArticlesTestingSet")

    for file_name in os.listdir(real_test_path):
        with open(os.path.join(real_test_path, file_name), "r", encoding="utf-8", errors="ignore") as handle:
            all_test_articles.append((handle.read(), 0))

    for file_name in os.listdir(fake_test_path):
        with open(os.path.join(fake_test_path, file_name), "r", encoding="utf-8", errors="ignore") as handle:
            all_test_articles.append((handle.read(), 1))

    return all_test_articles


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


def _to_tensor_dataset(articles: Sequence[Article], vocab: Dict[str, int], max_len: int = 300) -> TensorDataset:
    texts = [row[0] for row in articles]
    labels = [row[1] for row in articles]
    x_tensor = torch.tensor([_encode(t, vocab, max_len=max_len) for t in texts], dtype=torch.long)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def _evaluate(model: nn.Module, dataset: TensorDataset, device: torch.device, batch_size: int = 128) -> Dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    correct = 0
    total = 0
    tp = fp = fn = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            tp += ((preds == 1) & (batch_y == 1)).sum().item()
            fp += ((preds == 1) & (batch_y == 0)).sum().item()
            fn += ((preds == 0) & (batch_y == 1)).sum().item()

    accuracy = correct / total if total else 0.0
    precision_fake = tp / (tp + fp) if (tp + fp) else 0.0
    recall_fake = tp / (tp + fn) if (tp + fn) else 0.0
    fake_f1 = 0.0
    if precision_fake + recall_fake:
        fake_f1 = 2 * precision_fake * recall_fake / (precision_fake + recall_fake)

    return {
        "accuracy": accuracy,
        "fake_precision": precision_fake,
        "fake_recall": recall_fake,
        "fake_f1": fake_f1,
    }


def TrainAndEvaluateLSTM(all_articles: Sequence[Article], epochs: int = 15) -> float:
    random.seed(42)
    shuffled_articles = list(all_articles)
    random.shuffle(shuffled_articles)

    train_texts = [row[0] for row in shuffled_articles]
    vocab = _build_vocab(train_texts)

    train_dataset = _to_tensor_dataset(shuffled_articles, vocab)
    test_articles = _read_test_articles()
    test_dataset = _to_tensor_dataset(test_articles, vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    metrics = _evaluate(model, test_dataset, device)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Fake F1: {metrics['fake_f1']:.4f}")
    return metrics["accuracy"]


def TrainAndEvaluateLSTMWithAdaptiveSampling(
    real_articles: Sequence[Article],
    fake_articles: Sequence[Article],
    generated_fake_articles: Sequence[Article],
    epochs: int = 6,
    candidate_ratios: Sequence[int] = (1, 2, 3),
) -> Dict[str, float]:
    """
    RL-inspired adaptive sampling:
    for each epoch, choose fake:real ratio from candidate_ratios based on best recent validation fake-F1.
    """
    random.seed(42)

    real_articles = list(real_articles)
    fake_articles = list(fake_articles)
    generated_fake_articles = list(generated_fake_articles)

    random.shuffle(real_articles)
    random.shuffle(fake_articles)
    random.shuffle(generated_fake_articles)

    # Hold out validation from the original (non-generated) data.
    val_real_count = max(1, int(0.2 * len(real_articles)))
    val_fake_count = max(1, int(0.2 * len(fake_articles)))

    val_articles = real_articles[:val_real_count] + fake_articles[:val_fake_count]
    train_real = real_articles[val_real_count:]
    train_fake = fake_articles[val_fake_count:]

    train_vocab_articles = train_real + train_fake + generated_fake_articles
    vocab = _build_vocab([t for t, _ in train_vocab_articles])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    ratio_scores: Dict[int, List[float]] = defaultdict(list)

    val_dataset = _to_tensor_dataset(val_articles, vocab)

    for epoch in range(epochs):
        # choose ratio with highest recent score; fallback to 1:1
        if any(ratio_scores.values()):
            ratio = max(
                candidate_ratios,
                key=lambda r: (sum(ratio_scores[r]) / len(ratio_scores[r])) if ratio_scores[r] else -1.0,
            )
        else:
            ratio = candidate_ratios[0]

        real_pool = train_real.copy()
        random.shuffle(real_pool)
        fake_pool = (train_fake + generated_fake_articles).copy()
        random.shuffle(fake_pool)

        real_count = len(real_pool)
        fake_target = min(len(fake_pool), ratio * real_count)

        epoch_articles = real_pool + fake_pool[:fake_target]
        random.shuffle(epoch_articles)

        train_dataset = _to_tensor_dataset(epoch_articles, vocab)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = _evaluate(model, val_dataset, device)
        ratio_scores[ratio].append(val_metrics["fake_f1"])

        print(
            f"Epoch {epoch + 1}/{epochs} | ratio 1:{ratio} | "
            f"train_size={len(epoch_articles)} | loss={total_loss:.4f} | "
            f"val_fake_f1={val_metrics['fake_f1']:.4f}"
        )

    test_articles = _read_test_articles()
    test_dataset = _to_tensor_dataset(test_articles, vocab)
    test_metrics = _evaluate(model, test_dataset, device)

    # Which ratio ended up best by validation history
    best_ratio = max(
        candidate_ratios,
        key=lambda r: (sum(ratio_scores[r]) / len(ratio_scores[r])) if ratio_scores[r] else -1.0,
    )

    test_metrics["best_ratio"] = float(best_ratio)
    return test_metrics
