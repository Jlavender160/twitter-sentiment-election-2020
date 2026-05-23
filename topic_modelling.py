import os
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from wordcloud import WordCloud

BASE_DIR    = "/Users/james/Downloads/archive1"
TOPICS_DIR  = os.path.join(BASE_DIR, "breakdowns", "topics")
VIZ_DIR     = os.path.join(TOPICS_DIR, "visualizations")
MODEL_DIR   = os.path.join(TOPICS_DIR, "bertopic_model")
EMB_CACHE   = os.path.join(BASE_DIR, "topic_embeddings.npy")

BEFORE_CSV      = os.path.join(BASE_DIR, "tweets_before_election_sentiment.csv")
AFTER_CSV       = os.path.join(BASE_DIR, "tweets_after_election_sentiment.csv")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_TOPIC_SIZE  = 100
NR_TOPICS       = "auto"

_TWITTER_STOPWORDS = {
    'trump', 'donald', 'biden', 'joe', 'donaldtrump', 'joebiden',
    'rt', 'amp', 'http', 'https', 'co', 'www', 'tco',
    'would', 'could', 'get', 'got', 'also', 'really', 'even',
    'still', 'like', 'just', 'know', 'one', 'make', 'said',
    'say', 'will', 'going', 'us', 'u', 'im', 'ive', 'dont',
    'cant', 'via', 'new', 'people', 'time', 'year', 'today',
    'now', 'need', 'want', 'good', 'great', 'think', 'see',
    'election', 'vote', 'voted', 'voting', 'voter', 'president',
}


def build_stopwords():
    try:
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords', quiet=True)
        return list(set(stopwords.words('english')) | _TWITTER_STOPWORDS)
    except Exception:
        return list(_TWITTER_STOPWORDS)


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_data():
    print("Loading tweet data...")
    before_df = pd.read_csv(BEFORE_CSV)
    after_df  = pd.read_csv(AFTER_CSV)
    before_df["period"] = "before"
    after_df["period"]  = "after"
    combined = pd.concat([before_df, after_df], ignore_index=True)
    combined["created_at"] = pd.to_datetime(combined["created_at"], errors='coerce')
    combined = combined.dropna(subset=["clean_text"])
    combined["clean_text"] = combined["clean_text"].astype(str)
    print(f"  Before: {(combined['period'] == 'before').sum():,}  After: {(combined['period'] == 'after').sum():,}  Total: {len(combined):,}")
    return combined


def get_or_compute_embeddings(docs, embedding_model):
    if os.path.exists(EMB_CACHE):
        print(f"  Loading cached embeddings from {EMB_CACHE}")
        embeddings = np.load(EMB_CACHE)
        if embeddings.shape[0] == len(docs):
            return embeddings
        print("  Cache size mismatch — recomputing...")
    print(f"  Computing embeddings (~30-60 mins for 300K tweets, cached after first run)...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=256, device="mps")
    np.save(EMB_CACHE, embeddings)
    print(f"  Saved to {EMB_CACHE}")
    return embeddings


def build_topic_model():
    print("\nInitialising BERTopic...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric='cosine', random_state=42)

    hdbscan_model = HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, min_samples=10,
                             metric='euclidean', cluster_selection_method='eom',
                             prediction_data=True)

    vectorizer = CountVectorizer(stop_words=build_stopwords(),
                                 min_df=10, max_df=0.90, ngram_range=(1, 2))

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(embedding_model=embedding_model, umap_model=umap_model,
                           hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
                           ctfidf_model=ctfidf_model, nr_topics=NR_TOPICS, verbose=True)

    return topic_model, embedding_model


def compute_topic_distribution(df, group_cols):
    filtered = df[df["topic"] != -1]
    grp = filtered.groupby(group_cols + ["topic"]).size().reset_index(name="count")
    grp["pct"] = (grp["count"] / grp.groupby(group_cols)["count"].transform("sum") * 100).round(2)
    return grp.sort_values(group_cols + ["count"], ascending=False)


def save_breakdown_csvs(df, topic_model):
    os.makedirs(TOPICS_DIR, exist_ok=True)

    topic_info = topic_model.get_topic_info()
    topic_info[topic_info["Topic"] != -1].to_csv(
        os.path.join(TOPICS_DIR, "topic_keywords.csv"), index=False)

    save_cols = [c for c in ["candidate", "period", "sentiment", "gender",
                              "engagement_tier", "topic", "topic_label", "clean_text"]
                 if c in df.columns]
    df[save_cols].to_csv(os.path.join(TOPICS_DIR, "tweets_with_topics.csv"), index=False)

    for cols, fname in [
        (["candidate"],            "topic_distribution_by_candidate.csv"),
        (["period"],               "topic_distribution_by_period.csv"),
        (["candidate", "period"],  "topic_distribution_by_candidate_period.csv"),
        (["sentiment"],            "topic_distribution_by_sentiment.csv"),
    ]:
        compute_topic_distribution(df, cols).to_csv(os.path.join(TOPICS_DIR, fname), index=False)

    print(f"\nBreakdown CSVs saved to {TOPICS_DIR}/")


def save_word_clouds(topic_model, top_n=15):
    wc_dir = os.path.join(VIZ_DIR, "wordclouds")
    os.makedirs(wc_dir, exist_ok=True)
    valid_topics = topic_model.get_topic_info()
    valid_topics = valid_topics[valid_topics["Topic"] != -1]["Topic"].tolist()[:top_n]
    for tid in valid_topics:
        words = topic_model.get_topic(tid)
        if not words:
            continue
        wc = WordCloud(width=800, height=400, background_color='white',
                       max_words=30, colormap='viridis')
        wc.generate_from_frequencies({w: max(s, 0.001) for w, s in words})
        wc.to_file(os.path.join(wc_dir, f"topic_{tid}_wordcloud.png"))
    print(f"  Word clouds saved to {wc_dir}/")


def save_visualizations(topic_model, df, topics_over_time):
    os.makedirs(VIZ_DIR, exist_ok=True)
    try:
        topic_model.visualize_barchart(top_n_topics=20, n_words=8).write_html(
            os.path.join(VIZ_DIR, "topics_barchart.html"))
        topic_model.visualize_topics().write_html(
            os.path.join(VIZ_DIR, "topics_distance.html"))
        topic_model.visualize_heatmap().write_html(
            os.path.join(VIZ_DIR, "topics_heatmap.html"))
        if topics_over_time is not None:
            topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=15).write_html(
                os.path.join(VIZ_DIR, "topics_over_time.html"))
        print(f"  HTML visualizations saved to {VIZ_DIR}/")
    except Exception as e:
        print(f"  Visualization error (non-fatal): {e}")
    save_word_clouds(topic_model)


def print_summary(topic_model, df):
    topic_info = topic_model.get_topic_info()
    valid    = topic_info[topic_info["Topic"] != -1]
    outliers = topic_info[topic_info["Topic"] == -1]["Count"].values[0] if -1 in topic_info["Topic"].values else 0

    print(f"\n{'='*60}\nTOPIC MODELLING RESULTS\n{'='*60}")
    print(f"  Topics discovered:         {len(valid)}")
    print(f"  Outlier tweets (topic -1): {outliers:,}")
    print(f"  Tweets assigned to topics: {(df['topic'] != -1).sum():,}")

    print(f"\n  {'ID':<6} {'Size':>8}  Keywords")
    for _, row in valid.head(10).iterrows():
        kws = ", ".join([w for w, _ in topic_model.get_topic(row["Topic"])[:5]])
        print(f"  {row['Topic']:<6} {row['Count']:>8,}  {kws}")

    for period in ["before", "after"]:
        period_df = df[(df["period"] == period) & (df["topic"] != -1)]
        print(f"\n  {period.upper()}:")
        for tid, count in period_df["topic"].value_counts().head(5).items():
            kws = ", ".join([w for w, _ in topic_model.get_topic(tid)[:4]])
            print(f"    Topic {tid:<4} {count:>6,} ({count/len(period_df)*100:.1f}%)  [{kws}]")


def main():
    print("=" * 60)
    print("Twitter Topic Modelling — US Election 2020")
    print("=" * 60)

    df   = load_data()
    docs = [preprocess_text(t) for t in df["clean_text"].tolist()]

    topic_model, embedding_model = build_topic_model()
    embeddings = get_or_compute_embeddings(docs, embedding_model)

    print("\nFitting topic model...")
    topics, probs = topic_model.fit_transform(docs, embeddings)
    df["topic"] = topics

    topic_info = topic_model.get_topic_info().set_index("Topic")
    df["topic_label"] = df["topic"].map(
        lambda t: topic_info.loc[t, "Name"] if t in topic_info.index else "Outlier")

    print("\nComputing topics over time...")
    topics_over_time = None
    valid_ts = df["created_at"].notna()
    if valid_ts.sum() > 0:
        try:
            topics_over_time = topic_model.topics_over_time(
                [docs[i] for i in df[valid_ts].index],
                df.loc[valid_ts, "created_at"].tolist(), nr_bins=14)
            topics_over_time.to_csv(os.path.join(TOPICS_DIR, "topics_over_time.csv"), index=False)
            print("  topics_over_time.csv saved.")
        except Exception as e:
            print(f"  topics_over_time failed (non-fatal): {e}")

    os.makedirs(TOPICS_DIR, exist_ok=True)
    save_breakdown_csvs(df, topic_model)
    save_visualizations(topic_model, df, topics_over_time)

    print("\nSaving BERTopic model...")
    topic_model.save(MODEL_DIR)
    print(f"  Model saved to {MODEL_DIR}/")

    print_summary(topic_model, df)
    print(f"\n{'='*60}\nDONE!\n{'='*60}")


if __name__ == "__main__":
    main()
