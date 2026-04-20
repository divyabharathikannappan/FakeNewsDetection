# Alex Eliseev
# Data loading helpers for fake-news classification experiments.

from __future__ import annotations

import os
import random
from typing import List, Sequence, Tuple

Article = Tuple[str, int]


def _read_articles(folder_path: str, label: int, limit: int | None = None) -> List[Article]:
    articles = os.listdir(folder_path)
    random.shuffle(articles)
    if limit is not None:
        articles = articles[:limit]

    loaded: List[Article] = []
    for article in articles:
        file_path = os.path.join(folder_path, article)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read().replace("<br />", " ").replace("<br>", " ")
            loaded.append((text, label))
    return loaded


def _training_dirs() -> Tuple[str, str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    real_path = os.path.join(project_root, "Data", "train", "RealNewsArticlesTrainingSet")
    fake_path = os.path.join(project_root, "Data", "train", "FakeNewsArticlesTrainingSet")
    generated_path = os.path.join(
        project_root,
        "Data",
        "train",
        "FakeNewsArticlesArtificiallyGeneratedTrainingSet",
    )
    return real_path, fake_path, generated_path


def LoadTrainingArticleGroups(
    number_of_artificial_articles_to_use: int,
    use_all: int,
    number_of_real_fake_articles: int = 400,
    number_of_real_articles: int = 800,
) -> Tuple[List[Article], List[Article], List[Article]]:
    """
    Returns three independent groups:
      - real_articles (label=0)
      - original_fake_articles (label=1)
      - generated_fake_articles (label=1)
    """
    random.seed(42)

    real_path, fake_path, generated_path = _training_dirs()

    real_limit = None if use_all else number_of_real_articles
    fake_limit = None if use_all else number_of_real_fake_articles

    real_articles = _read_articles(real_path, label=0, limit=real_limit)
    fake_articles = _read_articles(fake_path, label=1, limit=fake_limit)
    generated_fake_articles = _read_articles(
        generated_path,
        label=1,
        limit=number_of_artificial_articles_to_use,
    )

    return real_articles, fake_articles, generated_fake_articles


def TrainAndEvaluateClassifer(
    NumberOfArtificialArticlesTouse: int,
    UseAll: int,
    NumberOfRealFakeArticles: int = 400,
    NumberOfRealArticles: int = 800,
) -> List[Article]:
    """
    Backwards-compatible loader used by existing scripts.
    Returns a single combined list of (text, label) tuples.
    """
    real_articles, fake_articles, generated_fake_articles = LoadTrainingArticleGroups(
        number_of_artificial_articles_to_use=NumberOfArtificialArticlesTouse,
        use_all=UseAll,
        number_of_real_fake_articles=NumberOfRealFakeArticles,
        number_of_real_articles=NumberOfRealArticles,
    )
    return real_articles + fake_articles + generated_fake_articles
