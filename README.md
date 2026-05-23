# Twitter Sentiment Analysis — 2020 US Presidential Election

**Final Year Project | BSc Computer Science | Nottingham Trent University**
**Module:** COMP30060 Final Year Project | **Year:** 3rd Year (2025–2026)
**Student:** James Lavender (N1212075)

---

## Abstract

This project applies transformer-based NLP (RoBERTa) to analyse sentiment in Twitter data surrounding the 2020 US Presidential Election. It investigates whether sentiment shifted after the election, whether Trump-related tweets were more negative than Biden-related tweets, and whether disinformation-linked topic prevalence increased post-election. Three formal hypotheses are tested using chi-square and z-tests across a dataset of approximately 1.7 million tweets.

---

## Research Questions / Hypotheses

| ID | Hypothesis | Result |
|----|-----------|--------|
| H1 | Sentiment distribution shifts significantly before vs after election day | **REJECTED** — χ²=6775, p≈0 |
| H2 | Trump-related tweets are significantly more negative than Biden-related tweets | **REJECTED** — χ²=24456, p≈0 (Trump 58.1% neg vs Biden 34.5% neg) |
| H3 | Disinformation-linked topics increase significantly after the election | **REJECTED** — z=−22.03, p≈0 (2.12% → 4.31%, +103%) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Sentiment model | `cardiffnlp/twitter-roberta-base-sentiment` (HuggingFace) |
| Topic modelling | BERTopic |
| Statistical tests | SciPy (`chi2_contingency`, `proportions_ztest`) |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Data handling | Pandas |
| Language | Python 3.10+ |

---

## Project Structure

```
├── cleantweets1.py              # Data cleaning pipeline (noise removal, normalisation)
├── sentiment_analysis.py        # RoBERTa-based sentiment classification
├── topic_modelling.py           # BERTopic topic discovery (~30 topics)
├── hypothesis_tests.py          # Chi-square (H1, H2) and z-test (H3) statistical tests
├── visualisations.py            # Chart generation (5 publication-quality figures)
├── manual_annotation_sample.py  # Exports 198-tweet sample for manual labelling
└── breakdowns/
    ├── charts/                  # Generated PNG charts
    └── topics/
        └── visualizations/      # BERTopic interactive HTML visualisations + wordclouds
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install transformers torch pandas scipy matplotlib seaborn bertopic plotly
```

### 2. Prepare the Dataset

Download the [2020 US Election Tweet dataset](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets) from Kaggle and place the CSV files in the project root.

### 3. Run the Pipeline (in order)

```bash
# Step 1: Clean the raw tweets
python cleantweets1.py

# Step 2: Run sentiment analysis (RoBERTa — GPU recommended)
python sentiment_analysis.py

# Step 3: Discover topics with BERTopic
python topic_modelling.py

# Step 4: Run hypothesis tests
python hypothesis_tests.py

# Step 5: Generate charts
python visualisations.py
```

> **Note:** `sentiment_analysis.py` processes ~1.7M tweets using a transformer model. A CUDA-capable GPU is strongly recommended. CPU runtime may exceed several hours.

---

## Key Results

- **Sentiment shift (H1):** Statistically significant shift in sentiment distribution before vs after election day (Cramér's V = 0.15, medium effect).
- **Candidate comparison (H2):** Trump-related tweets were significantly more negative (58.1% negative) compared to Biden-related tweets (34.5% negative). Cramér's V = 0.285 (medium effect).
- **Disinformation topics (H3):** Topics associated with electoral fraud/disinformation doubled in prevalence after the election (2.12% → 4.31%, +103% increase).
- **Top BERTopic topics discovered:** Kamala Harris, COVID-19, Pennsylvania, Georgia, "stop the count", concession, racism, evangelical voters, economy.

---

## Output Charts

All charts are saved to `breakdowns/charts/`:

| File | Description |
|------|-------------|
| `sentiment_before_after.png` | Sentiment distribution before vs after election day |
| `sentiment_by_candidate.png` | Sentiment split by Trump vs Biden mentions |
| `sentiment_candidate_period.png` | Candidate × period interaction |
| `sentiment_over_time.png` | Sentiment trend across the full timeline |
| `sentiment_shift.png` | Net sentiment shift visualisation |

---

## Author

**James Lavender** | N1212075 | Nottingham Trent University
BSc (Hons) Computer Science — Final Year Project 2025–2026
