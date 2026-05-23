"""
sentiment_analysis.py
-------------------------------------------------------------
WHAT THIS SCRIPT DOES:
1. Loads the Cardiff NLP twitter-roberta-base-sentiment-latest model
2. Classifies every tweet as positive, neutral, or negative
3. Infers author gender from Twitter usernames
4. Classifies tweets into high/low engagement tiers
5. Saves the labelled CSVs and generates sentiment breakdown tables
-------------------------------------------------------------
"""

import os
import re
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gender_guesser.detector as gender_detector

# --- Model settings ---
# Cardiff NLP RoBERTa pre-trained on Twitter data — chosen for domain match
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32    # tweets processed per batch
MAX_LENGTH = 128   # max tokens (sufficient for 280-char tweets)
DEVICE     = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
LABELS     = ["negative", "neutral", "positive"]  # label order from the model

BASE_DIR       = "/Users/james/Downloads/archive1"
BREAKDOWNS_DIR = os.path.join(BASE_DIR, "breakdowns")


# Dataset 

class TweetDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                             max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze()}


#  Sentiment inference 

def predict_sentiment(df, tokenizer, model, text_column="clean_text"):
    """Run batch inference through the transformer — returns predicted labels and confidence scores."""
    dataset    = TweetDataset(df[text_column].tolist(), tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_scores = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting sentiment"):
            probs = torch.softmax(
                model(input_ids=batch["input_ids"].to(DEVICE),
                      attention_mask=batch["attention_mask"].to(DEVICE)).logits,
                dim=1
            )
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    return all_preds, all_scores


def add_sentiment_columns(df, predictions, scores):
    """Append sentiment label and per-class probability columns to df."""
    df = df.copy()
    df["sentiment"]           = [LABELS[p] for p in predictions]
    df["sentiment_score_neg"] = [s[0] for s in scores]
    df["sentiment_score_neu"] = [s[1] for s in scores]
    df["sentiment_score_pos"] = [s[2] for s in scores]
    return df


# Gender prediction 

_gender_det = gender_detector.Detector(case_sensitive=False)

def _extract_first_name(screen_name):
    """Extract probable first name from Twitter screen name."""
    if not isinstance(screen_name, str):
        return None
    name  = re.sub(r'\d+$', '', screen_name.strip())
    parts = re.split(r'[_.\\-]', name)
    for part in parts:
        if len(part) >= 3 and part.isalpha():
            return part
    for part in re.sub(r'([A-Z][a-z]+)', r' \1', name).split():
        if len(part) >= 3 and part.isalpha():
            return part
    return None

def predict_gender(screen_name):
    """Return 'male', 'female', or 'unknown' from a Twitter screen name."""
    first = _extract_first_name(screen_name)
    if not first:
        return "unknown"
    result = _gender_det.get_gender(first)
    if result in ("male", "mostly_male"):
        return "male"
    if result in ("female", "mostly_female"):
        return "female"
    return "unknown"

def add_gender_column(df):
    df = df.copy()
    df["gender"] = df["user_screen_name"].apply(predict_gender)
    return df


# Engagement tier 

def add_engagement_tier(df):
    """Classify tweets into high (top 25%) / low engagement based on likes + retweets."""
    df = df.copy()
    df["engagement"] = df["likes"].fillna(0) + df["retweet_count"].fillna(0)
    threshold = df["engagement"].quantile(0.75)
    df["engagement_tier"] = df["engagement"].apply(lambda x: "high" if x >= threshold else "low")
    return df


#  Breakdown helper 

def compute_breakdown(df, group_cols):
    """Sentiment counts + percentages grouped by group_cols."""
    if group_cols:
        grp = df.groupby(group_cols + ["sentiment"]).size().unstack(fill_value=0)
    else:
        grp = df.groupby("sentiment").size().to_frame().T
        grp.index = ["all"]
    for col in LABELS:
        if col not in grp.columns:
            grp[col] = 0
    grp["total"]        = grp[LABELS].sum(axis=1)
    grp["negative_pct"] = (grp["negative"] / grp["total"] * 100).round(2)
    grp["neutral_pct"]  = (grp["neutral"]  / grp["total"] * 100).round(2)
    grp["positive_pct"] = (grp["positive"] / grp["total"] * 100).round(2)
    return grp[LABELS + ["total", "negative_pct", "neutral_pct", "positive_pct"]].reset_index()


def save_breakdowns(combined_df):
    """Save all sentiment breakdown CSVs to breakdowns/."""
    os.makedirs(BREAKDOWNS_DIR, exist_ok=True)
    breakdowns = {
        "sentiment_overall.csv":                  [],
        "sentiment_by_candidate.csv":             ["candidate"],
        "sentiment_by_period.csv":                ["period"],
        "sentiment_by_candidate_period.csv":      ["candidate", "period"],
        "sentiment_by_gender.csv":                ["gender"],
        "sentiment_by_engagement.csv":            ["engagement_tier"],
    }
    for fname, cols in breakdowns.items():
        compute_breakdown(combined_df, cols).to_csv(
            os.path.join(BREAKDOWNS_DIR, fname), index=False)

    # Country / state — drop nulls first
    compute_breakdown(combined_df.dropna(subset=["country"]), ["country"])\
        .sort_values("total", ascending=False).head(20)\
        .to_csv(os.path.join(BREAKDOWNS_DIR, "sentiment_by_country.csv"), index=False)
    compute_breakdown(combined_df.dropna(subset=["state"]), ["state"])\
        .sort_values("total", ascending=False)\
        .to_csv(os.path.join(BREAKDOWNS_DIR, "sentiment_by_state.csv"), index=False)

    print(f"\nBreakdown CSVs saved to: {BREAKDOWNS_DIR}/")


# --- Summary print ---

def print_summary(df, label):
    print(f"\n{'='*50}\n{label} — Sentiment Distribution\n{'='*50}")
    dist = df["sentiment"].value_counts(normalize=True).round(3)
    for sent, pct in dist.items():
        print(f"  {sent:10s}: {pct*100:5.1f}%")
    print("\nBy Candidate:")
    for cand in df["candidate"].unique():
        print(f"\n  {cand.upper()}:")
        for sent, pct in df[df["candidate"] == cand]["sentiment"].value_counts(normalize=True).round(3).items():
            print(f"    {sent:10s}: {pct*100:5.1f}%")


# --- Main ---

def main():
    # STEP 1: Load the Cardiff NLP RoBERTa transformer onto the MPS/GPU device
    print(f"\nLoading model on device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    print("Model loaded.")

    # STEP 2: Load the cleaned tweet CSVs produced by cleantweets1.py
    print("\nLoading tweet data...")
    before_df = pd.read_csv(os.path.join(BASE_DIR, "tweets_before_election.csv"))
    after_df  = pd.read_csv(os.path.join(BASE_DIR, "tweets_after_election.csv"))
    print(f"  Before: {len(before_df):,}  After: {len(after_df):,}")

    # STEP 3: Classify every tweet as positive, neutral, or negative
    print("\n--- Sentiment: BEFORE ---")
    before_df = add_sentiment_columns(before_df, *predict_sentiment(before_df, tokenizer, model))
    print("\n--- Sentiment: AFTER ---")
    after_df  = add_sentiment_columns(after_df,  *predict_sentiment(after_df,  tokenizer, model))

    # STEP 4: Infer gender from Twitter usernames using the gender_guesser library
    print("\nPredicting gender...")
    before_df = add_gender_column(before_df)
    after_df  = add_gender_column(after_df)

    # Drop usernames after gender inference — data minimisation (GDPR)
    before_df = before_df.drop(columns=["user_screen_name"])
    after_df  = after_df.drop(columns=["user_screen_name"])

    # STEP 5: Label each tweet as high or low engagement (top 25% of likes + retweets = high)
    print("Computing engagement tiers...")
    before_df = add_engagement_tier(before_df)
    after_df  = add_engagement_tier(after_df)

    # STEP 6: Save the labelled datasets to CSV
    before_out = os.path.join(BASE_DIR, "tweets_before_election_sentiment.csv")
    after_out  = os.path.join(BASE_DIR, "tweets_after_election_sentiment.csv")
    before_df.to_csv(before_out, index=False)
    after_df.to_csv(after_out,  index=False)
    print(f"\nSaved: {before_out}\nSaved: {after_out}")

    print_summary(before_df, "BEFORE Election (Oct 27 – Nov 2, 2020)")
    print_summary(after_df,  "AFTER Election  (Nov 3 – Nov 8, 2020)")

    before_df["period"] = "before"
    after_df["period"]  = "after"
    combined = pd.concat([before_df, after_df], ignore_index=True)

    print("\nGenerating breakdowns...")
    save_breakdowns(combined)

    before_pos = (before_df["sentiment"] == "positive").mean()
    after_pos  = (after_df["sentiment"]  == "positive").mean()
    before_neg = (before_df["sentiment"] == "negative").mean()
    after_neg  = (after_df["sentiment"]  == "negative").mean()
    print(f"\nPositive: {before_pos*100:.1f}% → {after_pos*100:.1f}% ({(after_pos-before_pos)*100:+.1f}%)")
    print(f"Negative: {before_neg*100:.1f}% → {after_neg*100:.1f}% ({(after_neg-before_neg)*100:+.1f}%)")
    print("\nDone!")


if __name__ == "__main__":
    main()
