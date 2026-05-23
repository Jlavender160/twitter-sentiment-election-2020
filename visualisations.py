"""
visualisations.py
-------------------------------------------------------------
WHAT THIS SCRIPT DOES:
Reads the sentiment breakdown CSVs produced by sentiment_analysis.py
and generates 6 publication-ready charts saved to breakdowns/charts/.

Charts produced:
    1. sentiment_before_after.png     — Overall sentiment before vs after election
    2. sentiment_by_candidate.png     — Trump vs Biden sentiment comparison
    3. sentiment_candidate_period.png — Candidate × period (4-group stacked bar)
    4. sentiment_over_time.png        — Daily sentiment % across the 14-day window
    5. sentiment_shift.png            — Change in sentiment % before → after
    6. sentiment_by_gender.png        — Sentiment split by inferred author gender
-------------------------------------------------------------
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_DIR   = "/Users/james/Downloads/archive1"
BREAKS_DIR = os.path.join(BASE_DIR, "breakdowns")
CHARTS_DIR = os.path.join(BREAKS_DIR, "charts")

COLOURS = {"negative": "#d9534f", "neutral": "#f0ad4e", "positive": "#5cb85c"}
PERIOD_LABELS = {"before": "Before (Oct 27 – Nov 2)", "after": "After (Nov 3 – Nov 9)"}
SENTIMENTS = ["negative", "neutral", "positive"]
PCT_COLS   = ["negative_pct", "neutral_pct", "positive_pct"]

os.makedirs(CHARTS_DIR, exist_ok=True)


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _grouped_bar(ax, df, x_labels, width=0.25, ylim=75):
    """Draw a grouped bar chart on ax from a DataFrame with PCT_COLS columns."""
    x = np.arange(len(x_labels))
    for i, (sent, col) in enumerate(zip(SENTIMENTS, PCT_COLS)):
        vals = df[col].values
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=sent.capitalize(), color=COLOURS[sent], alpha=0.88, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Percentage of tweets (%)", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)


# --- Chart 1: Overall sentiment before vs after ---

def chart_before_after():
    df = pd.read_csv(os.path.join(BREAKS_DIR, "sentiment_by_period.csv")).set_index("period").loc[["before", "after"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar(ax, df, [PERIOD_LABELS["before"], PERIOD_LABELS["after"]])
    ax.set_title("Overall Tweet Sentiment — Before vs After the 2020 US Election",
                 fontsize=12, fontweight='bold', pad=12)
    _save(os.path.join(CHARTS_DIR, "sentiment_before_after.png"))


# --- Chart 2: Trump vs Biden overall sentiment ---

def chart_candidate():
    df = pd.read_csv(os.path.join(BREAKS_DIR, "sentiment_by_candidate.csv")).set_index("candidate").loc[["trump", "biden"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(ax, df, ["Trump", "Biden"])
    ax.set_title("Tweet Sentiment by Candidate — Trump vs Biden",
                 fontsize=12, fontweight='bold', pad=12)
    _save(os.path.join(CHARTS_DIR, "sentiment_by_candidate.png"))


# --- Chart 3: Stacked bar — candidate × period ---

def chart_candidate_period():
    df = pd.read_csv(os.path.join(BREAKS_DIR, "sentiment_by_candidate_period.csv"))
    order = [("trump", "before"), ("trump", "after"), ("biden", "before"), ("biden", "after")]
    labels = ["Trump\nBefore", "Trump\nAfter", "Biden\nBefore", "Biden\nAfter"]

    neg, neu, pos = [], [], []
    for cand, period in order:
        row = df[(df["candidate"] == cand) & (df["period"] == period)].iloc[0]
        neg.append(row["negative_pct"])
        neu.append(row["neutral_pct"])
        pos.append(row["positive_pct"])

    x, width = np.arange(4), 0.55
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x, neg, width, label="Negative", color=COLOURS["negative"], alpha=0.88)
    b2 = ax.bar(x, neu, width, bottom=neg, label="Neutral", color=COLOURS["neutral"], alpha=0.88)
    b3 = ax.bar(x, pos, width, bottom=[n + nu for n, nu in zip(neg, neu)],
                label="Positive", color=COLOURS["positive"], alpha=0.88)

    for bars, vals, bases in [(b1, neg, [0]*4), (b2, neu, neg),
                               (b3, pos, [n + nu for n, nu in zip(neg, neu)])]:
        for bar, v, base in zip(bars, vals, bases):
            if v > 4:
                ax.text(bar.get_x() + bar.get_width() / 2, base + v / 2,
                        f"{v:.1f}%", ha='center', va='center', fontsize=7.5,
                        color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Percentage of tweets (%)", fontsize=10)
    ax.set_title("Tweet Sentiment by Candidate and Period", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, 108)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    _save(os.path.join(CHARTS_DIR, "sentiment_candidate_period.png"))


# --- Chart 4: Sentiment over time (daily line chart) ---

def chart_over_time():
    before_file = os.path.join(BASE_DIR, "tweets_before_election_sentiment.csv")
    after_file  = os.path.join(BASE_DIR, "tweets_after_election_sentiment.csv")
    if not os.path.exists(before_file) or not os.path.exists(after_file):
        print("  Skipping over-time chart — sentiment CSVs not found.")
        return

    print("  Loading sentiment CSVs for time-series chart...")
    cols = ["created_at", "sentiment"]
    combined = pd.concat([pd.read_csv(before_file, usecols=cols),
                          pd.read_csv(after_file,  usecols=cols)], ignore_index=True)
    combined["created_at"] = pd.to_datetime(combined["created_at"], errors="coerce")
    combined = combined.dropna(subset=["created_at"])
    combined["date"] = combined["created_at"].dt.date

    daily = combined.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    daily["total"] = daily.sum(axis=1)
    for sent in SENTIMENTS:
        if sent in daily.columns:
            daily[f"{sent}_pct"] = daily[sent] / daily["total"] * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    for sent in SENTIMENTS:
        col = f"{sent}_pct"
        if col in daily.columns:
            ax.plot(daily.index, daily[col], label=sent.capitalize(),
                    color=COLOURS[sent], linewidth=2, marker='o', markersize=3)

    election_day = datetime.date(2020, 11, 3)
    ax.axvline(x=election_day, color='black', linestyle='--', linewidth=1.2, label='Election Day (Nov 3)')
    ax.text(election_day, ax.get_ylim()[1] * 0.95, ' Election\n Day', fontsize=8, va='top')
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Percentage of daily tweets (%)", fontsize=10)
    ax.set_title("Daily Tweet Sentiment — US Election 2020 (Oct 27 – Nov 9)",
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    _save(os.path.join(CHARTS_DIR, "sentiment_over_time.png"))


# --- Chart 6: Gender sentiment pie charts ---

def chart_gender():
    df = pd.read_csv(os.path.join(BREAKS_DIR, "sentiment_by_gender.csv"))
    df = df[df["gender"].isin(["male", "female"])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colours = [COLOURS["negative"], COLOURS["neutral"], COLOURS["positive"]]

    for ax, (_, row) in zip(axes, df.iterrows()):
        vals = [row["negative"], row["neutral"], row["positive"]]
        ax.pie(vals, labels=["Negative", "Neutral", "Positive"],
               colors=colours, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{row['gender'].capitalize()} (n={int(row['total']):,})", fontsize=12)

    plt.suptitle("Sentiment Distribution by Inferred Gender", fontsize=13, fontweight="bold")
    _save(os.path.join(CHARTS_DIR, "sentiment_by_gender.png"))


# --- Chart 5: Diverging bar — sentiment shift before → after ---

def chart_shift():
    df = pd.read_csv(os.path.join(BREAKS_DIR, "sentiment_by_candidate_period.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, candidate in zip(axes, ["trump", "biden"]):
        subset = df[df["candidate"] == candidate].set_index("period")
        delta  = subset.loc["after", PCT_COLS].values - subset.loc["before", PCT_COLS].values
        colours = [COLOURS[s] for s in SENTIMENTS]
        y = np.arange(3)
        bars = ax.barh(y, delta, color=colours, alpha=0.85, edgecolor='white', height=0.5)
        for bar, d in zip(bars, delta):
            xpos = d + (0.3 if d >= 0 else -0.3)
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{d:+.1f}%", ha='left' if d >= 0 else 'right', va='center', fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([s.capitalize() for s in SENTIMENTS], fontsize=10)
        ax.set_xlabel("Change in sentiment % (After − Before)", fontsize=9)
        ax.set_title(f"{candidate.capitalize()} tweets — sentiment shift", fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    plt.suptitle("Sentiment Shift Before → After the 2020 US Election (Δ%)",
                 fontsize=12, fontweight='bold', y=1.02)
    _save(os.path.join(CHARTS_DIR, "sentiment_shift.png"))


# --- Main ---

def main():
    print("=" * 60)
    print("  Visualisations — Twitter Sentiment 2020 US Election")
    print(f"  Output: {CHARTS_DIR}")
    print("=" * 60)
    print("\n[1/5] Overall sentiment before vs after...")
    chart_before_after()
    print("\n[2/5] Sentiment by candidate...")
    chart_candidate()
    print("\n[3/5] Sentiment by candidate × period...")
    chart_candidate_period()
    print("\n[4/5] Sentiment over time...")
    chart_over_time()
    print("\n[5/5] Sentiment shift...")
    chart_shift()
    print("\n[6/6] Gender sentiment pie charts...")
    chart_gender()
    print(f"\n{'='*60}\n  All charts saved to: {CHARTS_DIR}\n  DONE\n{'='*60}")


if __name__ == "__main__":
    main()
