from __future__ import annotations

from TrainAndEvaluateClassifier import LoadTrainingArticleGroups, TrainAndEvaluateClassifer
from TrainAndEvaluateLSTM import TrainAndEvaluateLSTM, TrainAndEvaluateLSTMWithAdaptiveSampling


def run_three_experiments():
    print("=== Fake News Detection: 3-Setting Reproduction ===\n")

    # Shared load settings (change use_all to 1 for full run).
    use_all = 0
    generated_count = 200

    real_articles, fake_articles, generated_fake_articles = LoadTrainingArticleGroups(
        number_of_artificial_articles_to_use=generated_count,
        use_all=use_all,
        number_of_real_fake_articles=400,
        number_of_real_articles=800,
    )

    baseline_data = real_articles + fake_articles
    generated_aug_data = baseline_data + generated_fake_articles

    print("1) Baseline (original real + original fake)")
    baseline_acc = TrainAndEvaluateLSTM(baseline_data, epochs=6)

    print("\n2) Baseline + generated fake samples")
    generated_aug_acc = TrainAndEvaluateLSTM(generated_aug_data, epochs=6)

    print("\n3) Baseline + generated fake + RL-inspired adaptive sampling")
    adaptive_metrics = TrainAndEvaluateLSTMWithAdaptiveSampling(
        real_articles=real_articles,
        fake_articles=fake_articles,
        generated_fake_articles=generated_fake_articles,
        epochs=6,
        candidate_ratios=(1, 2, 3),
    )

    print("\n=== Experiment Summary ===")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"+Generated accuracy: {generated_aug_acc:.4f}")
    print(
        "+Generated + adaptive sampling "
        f"accuracy: {adaptive_metrics['accuracy']:.4f}, "
        f"fake_f1: {adaptive_metrics['fake_f1']:.4f}, "
        f"best_ratio: 1:{int(adaptive_metrics['best_ratio'])}"
    )


def run_legacy_single_experiment():
    print("=== Running Single Model (Legacy Baseline) ===")
    data = TrainAndEvaluateClassifer(NumberOfArtificialArticlesTouse=50, UseAll=1)
    accuracy = TrainAndEvaluateLSTM(data)
    print(f"\nLegacy Baseline Accuracy (50 synthetic): {accuracy:.4f}")


if __name__ == "__main__":
    print("Starting Project...\n")
    run_three_experiments()
    print("\n=== Done ===")
