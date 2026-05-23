
import os
import sys
import argparse
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR    = "/Users/james/Downloads/archive1"
BREAKS_DIR  = os.path.join(BASE_DIR, "breakdowns")
SAMPLE_FILE = os.path.join(BREAKS_DIR, "manual_annotation_sample.csv")
CHARTS_DIR  = os.path.join(BREAKS_DIR, "charts")

BEFORE_CSV = os.path.join(BASE_DIR, "tweets_before_election_sentiment.csv")
AFTER_CSV  = os.path.join(BASE_DIR, "tweets_after_election_sentiment.csv")

SAMPLE_SIZE  = 200
RANDOM_SEED  = 42
LABEL_COLUMN = "manual_label"
VALID_LABELS = {"negative", "neutral", "positive"}


# ─── Stage 1: Export sample for manual labelling ─────────────────────────────

def export_sample():
    print("=" * 60)
    print("  STAGE 1 — Exporting annotation sample")
    print("=" * 60)

    before_df = pd.read_csv(BEFORE_CSV, usecols=["id", "created_at", "candidate",
                                                   "clean_text", "sentiment"])
    after_df  = pd.read_csv(AFTER_CSV,  usecols=["id", "created_at", "candidate",
                                                   "clean_text", "sentiment"])
    before_df["period"] = "before"
    after_df["period"]  = "after"

    combined = pd.concat([before_df, after_df], ignore_index=True)
    combined = combined.dropna(subset=["clean_text"])
    combined = combined[combined["clean_text"].str.strip() != ""]

    # Stratified sample: equal split across sentiment classes for representativeness
    n_per_class = SAMPLE_SIZE // 3  # 66 per class (198 total — last 2 from largest class)
    sampled_parts = []
    for sentiment in ["negative", "neutral", "positive"]:
        subset = combined[combined["sentiment"] == sentiment]
        n      = min(n_per_class, len(subset))
        sampled_parts.append(subset.sample(n=n, random_state=RANDOM_SEED))

    sample = pd.concat(sampled_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)

    # Add empty manual_label column for James to fill in
    sample[LABEL_COLUMN] = ""

    # Keep only columns needed for annotation
    out_cols = ["id", "created_at", "candidate", "period", "clean_text",
                "sentiment", LABEL_COLUMN]
    sample = sample[out_cols].reset_index(drop=True)

    os.makedirs(BREAKS_DIR, exist_ok=True)
    sample.to_csv(SAMPLE_FILE, index=False)

    print(f"\n  Sample size:    {len(sample)} tweets")
    print(f"  Saved to:       {SAMPLE_FILE}")
    print(f"\n  Sentiment distribution in sample:")
    print(sample["sentiment"].value_counts().to_string())

    print(f"""

  
""")


# ─── Stage 2: Evaluate manual labels vs model predictions ───────────────────

def evaluate():
    print("=" * 60)
    print("  STAGE 2 — Evaluating manual labels vs model predictions")
    print("=" * 60)

    if not os.path.exists(SAMPLE_FILE):
        print(f"\n  ERROR: {SAMPLE_FILE} not found.")
        print("  Run Stage 1 first: python3 manual_annotation_sample.py")
        sys.exit(1)

    df = pd.read_csv(SAMPLE_FILE)

    # Drop unlabelled rows — evaluate on labelled subset only
    df = df[df[LABEL_COLUMN].notna() & (df[LABEL_COLUMN].str.strip() != "")].copy()
    print(f"\n  Evaluating on {len(df)} labelled rows.")

    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip().str.lower()
    df["sentiment"]  = df["sentiment"].str.strip().str.lower()

    # Check all manual labels are valid
    invalid = df[~df[LABEL_COLUMN].isin(VALID_LABELS)]
    if len(invalid) > 0:
        print(f"\n  ERROR: Invalid labels found in '{LABEL_COLUMN}':")
        print(invalid[[LABEL_COLUMN]].value_counts().to_string())
        print(f"  Valid values: {VALID_LABELS}")
        sys.exit(1)

    y_true = df[LABEL_COLUMN].tolist()  # Human labels
    y_pred = df["sentiment"].tolist()   # Model predictions

    # ── Metrics ──
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   target_names=["negative", "neutral", "positive"],
                                   digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=["negative", "neutral", "positive"])

    print(f"\n  Sample size: {len(df)} tweets")
    print(f"  Accuracy:    {acc:.4f}  ({acc*100:.1f}%)\n")
    print("  Classification Report:")
    print(report)

    # ── Confusion matrix ──
    print("  Confusion Matrix (rows = manual label, cols = model prediction):")
    cm_df = pd.DataFrame(cm,
                         index=["True Neg", "True Neu", "True Pos"],
                         columns=["Pred Neg", "Pred Neu", "Pred Pos"])
    print(cm_df.to_string())

    # Save confusion matrix chart
    os.makedirs(CHARTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Negative", "Neutral", "Positive"])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title("Confusion Matrix — Manual Annotation vs CardiffNLP Model", pad=12)
    cm_path = os.path.join(CHARTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Confusion matrix chart saved: {cm_path}")

    # ── Error analysis ──
    errors = df[df[LABEL_COLUMN] != df["sentiment"]].copy()
    errors["error_type"] = errors[LABEL_COLUMN] + " → " + errors["sentiment"]
    print(f"\n  Error Analysis:")
    print(f"  Total disagreements: {len(errors)} / {len(df)} ({100*len(errors)/len(df):.1f}%)")
    print(f"\n  Most common error types:")
    print(errors["error_type"].value_counts().head(6).to_string())

    error_path = os.path.join(BREAKS_DIR, "manual_annotation_errors.csv")
    errors[["id", "candidate", "period", "clean_text", LABEL_COLUMN, "sentiment", "error_type"]]\
        .to_csv(error_path, index=False)
    print(f"\n  Error details saved: {error_path}")

    # ── Save results summary ──
    results_path = os.path.join(BREAKS_DIR, "manual_annotation_results.txt")
    with open(results_path, "w") as f:
        f.write("Manual Annotation Evaluation — CardiffNLP Twitter-RoBERTa Baseline\n")
        f.write(f"Sample size: {len(df)} tweets\n")
        f.write(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
    print(f"\n  Results saved: {results_path}")

    print(f"\n{'='*60}")
    print("  DONE")
    print("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual annotation baseline for dissertation.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Stage 2: evaluate manual labels vs model predictions")
    args = parser.parse_args()

    if args.evaluate:
        evaluate()
    else:
        export_sample()
