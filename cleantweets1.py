# cleantweets1.py — Tweet Cleaning Pipeline
# ---------------------------------------------------------------
# WHAT THIS SCRIPT DOES:
# Takes two raw Kaggle CSV files (Trump + Biden hashtag tweets)
# and runs a 15-stage cleaning pipeline on each.
# Output: tweets_before_election.csv and tweets_after_election.csv
# ---------------------------------------------------------------

import pandas as pd
import re
import html
from datetime import datetime

# --- Settings: election date and time window ---
ELECTION_DATE   = datetime(2020, 11, 3)  # US Election Day
DAYS_BEFORE     = 7   # keep tweets from 7 days before the election
DAYS_AFTER      = 7   # keep tweets from 7 days after the election
MIN_TEXT_LENGTH = 20  # discard tweets shorter than 20 characters after cleaning
TRUMP_FINAL     = 'hashtag_donaldtrump_final.csv'
BIDEN_FINAL     = 'hashtag_joebiden_final.csv'

# --- Regex patterns used for text cleaning ---
# These are compiled once upfront for performance across millions of rows
_HTML_ENTITY_RE     = re.compile(r'&\w+;|&\d+;?|&#\d+;?|&[a-zA-Z]+\d*;?')
_URL_PATTERNS       = re.compile(
    r'https?:?/?/?[\s]*\S*|www\.\S+|\S+\.com\S*|\S+\.co/\S*'
    r'|\S+\.ly/\S*|\S+\.be/\S*|\S+\.kr/\S*|://\S*|ttps://\S*'
)
_MENTION_RE         = re.compile(r'@[A-Za-z0-9_]+')
_HASHTAG_WORD_RE    = re.compile(r'#(\w+)')
_RT_RE              = re.compile(r'^RT\s*:?\s*', re.IGNORECASE)
_DOTS_RE            = re.compile(r'\.{2,}')
_CAMEL_LOWER_UPPER  = re.compile(r'([a-z])([A-Z])')
_CAMEL_ALPHA_DIGIT  = re.compile(r'([a-zA-Z])(\d)')
_CAMEL_DIGIT_ALPHA  = re.compile(r'(\d)([a-zA-Z])')
_WHITESPACE_RE      = re.compile(r'\s+')
_AMP_RE             = re.compile(r'&\w*\d*;?')
_EMOJI_RE           = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937\U00010000-\U0010ffff\u2640-\u2642"
    "\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "\u2022\u2764\u2665\u25b6\u25cf\u2714\u2611\u2610"
    "\u261d\u270a-\u270d\u2728\u2b50\u26a0\u203c\u2049"
    "]+",
    flags=re.UNICODE
)
_SPAM_RE            = re.compile(r'click here|buy now|free gift|limited time|►+|⬇+|giveaway|win a', re.IGNORECASE)
_HASHTAG_COUNT_RE   = re.compile(r'#\w+')
_BOT_SCREEN_RE      = re.compile(r'\d{6,}')
_NON_ENGLISH_RE     = re.compile(
    r'\bdie\b|\bder\b|\bdas\b|\bund\b|\bist\b|\bvon\b'
    r'|\bque\b|\bdel\b|\blos\b|\blas\b|\bpor\b|\buna\b'
    r'|\ble\b|\bla\b|\bles\b|\bdes\b|\best\b'
    r'|\bdi\b|\bche\b|\bnon\b|\bper\b|\bil\b'
    r'|\bem\b|\bpara\b|\bcom\b|\bnao\b|\buma\b'
    r'|\bwie\b|\bwre\b|\bnunmehr\b|\bgeben\b'
)
_ENGLISH_WORDS = {
    'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall',
    'this', 'that', 'these', 'those', 'what', 'which', 'who',
    'and', 'but', 'or', 'for', 'with', 'about', 'against',
    'not', 'you', 'your', 'they', 'them', 'their', 'our', 'his', 'her',
    'trump', 'biden', 'vote', 'election', 'president', 'america'
}
_FUNCTION_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'i', 'we', 'you', 'he', 'she', 'they',
    'it', 'this', 'that', 'not', 'no', 'so', 'but', 'and', 'or',
    'to', 'of', 'in', 'on', 'at', 'for', 'with', 'my', 'your',
    'his', 'her', 'our', 'its', 'if', 'just', 'get', 'go', 'out',
    'im', 'ive', 'id', 'ill', 'weve', 'theyre', 'dont', 'cant',
    'wont', 'isnt', 'wasnt', 'hes', 'shes', 'thats', 'whats'
}
_NORM_RE            = re.compile(r'[^\w\s]')
_TRAILING_VIA_RE    = re.compile(r'\s+via\s*$', re.IGNORECASE)


# Text cleaning 

def clean_text(text):
    """Remove URLs, mentions, hashtag symbols, RT prefix, CamelCase split, normalise whitespace."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = _HTML_ENTITY_RE.sub(' ', text)
    text = _URL_PATTERNS.sub('', text)
    text = _MENTION_RE.sub('', text)
    text = _HASHTAG_WORD_RE.sub(r'\1', text)
    text = text.replace('#', '')
    text = _RT_RE.sub('', text)
    text = _DOTS_RE.sub('', text)
    text = _CAMEL_LOWER_UPPER.sub(r'\1 \2', text)
    text = _CAMEL_ALPHA_DIGIT.sub(r'\1 \2', text)
    text = _CAMEL_DIGIT_ALPHA.sub(r'\1 \2', text)
    return _WHITESPACE_RE.sub(' ', text).strip()


def remove_emojis(text):
    """Remove emojis and non-ASCII characters."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = _EMOJI_RE.sub('', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return _WHITESPACE_RE.sub(' ', text).strip()


def final_clean(text):
    """Final pass to catch surviving @mentions, URL fragments, and trailing 'via'."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = _MENTION_RE.sub('', text)
    text = _AMP_RE.sub('', text)
    text = _URL_PATTERNS.sub('', text)
    text = _TRAILING_VIA_RE.sub('', text)
    return _WHITESPACE_RE.sub(' ', text).strip()


# Quality filters 

def detect_spam(text):
    """Detect promotional phrases or excessive hashtags (>5)."""
    text = str(text)
    return bool(_SPAM_RE.search(text)) or len(_HASHTAG_COUNT_RE.findall(text)) > 5


def detect_bot_account(row):
    """Detect bots by numeric screen name (6+ digits) or zero followers."""
    if _BOT_SCREEN_RE.search(str(row.get('user_screen_name', ''))):
        return True
    followers = row.get('user_followers_count', 0)
    return pd.isna(followers) or followers == 0


def is_english(text):
    """Lightweight English filter: ASCII ratio, non-English word check, English word count."""
    if not text or len(text) < 15:
        return False
    if sum(1 for c in text if ord(c) < 128) / len(text) < 0.9:
        return False
    text_lower = text.lower()
    if _NON_ENGLISH_RE.search(text_lower):
        return False
    padded = ' ' + text_lower + ' '
    matches = sum(1 for w in _ENGLISH_WORDS if f' {w} ' in padded)
    words = text.split()
    if words and matches / len(words) < 0.15:
        return False
    return matches >= 3


def detect_low_quality(text):
    """Reject template tweets, hashtag soup, keyword lists, and very short tweets."""
    if pd.isna(text) or not text:
        return True
    text = str(text)
    if re.search(r'_{2,}', text) or re.search(r'_\s*\.', text):
        return True
    if text[0].islower():
        return True
    if re.search(r'[=\-]{3,}', text):
        return True
    if re.search(r'\b[A-Z]\s[A-Z]\s[A-Z]\s[A-Z]', text):
        return True
    if re.search(r'\w+:\s*["\']?\w+', text) and text.count(':') >= 2:
        return True
    words = text.split()
    if len(words) < 4:
        return True
    capitalized = sum(1 for w in words if len(w) > 1 and w[0].isupper())
    if capitalized / len(words) > 0.7 and sum(1 for w in words if w[0].islower()) < 2:
        return True
    if '$' in text:
        return True
    if re.search(r'(list|name|what|pick).{0,20}(your|you|a)\b', text, re.IGNORECASE):
        if '?' in text or '___' in text or 'comment' in text.lower():
            return True
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) >= 4 and sum(1 for s in sentences if len(s.split()) <= 2) >= 3:
        return True
    words_lower = {w.lower().strip('!?,."\'') for w in words}
    if not words_lower & _FUNCTION_WORDS:
        return True
    return False


#  Per-user near-duplicate removal 

def remove_per_user_near_duplicates(df):
    """Remove tweets where the same user posted the same core text with different openers."""
    df['_user_norm'] = df['clean_text'].str.lower().str.replace(_NORM_RE, '', regex=True)
    df = df.drop_duplicates(subset=['user_screen_name', '_user_norm'])
    return df.drop(columns=['_user_norm']).reset_index(drop=True)


#  Main 

def process_csv(input_file, output_file):
    """Run all 15 cleaning stages on one candidate's raw CSV and save the result."""
    print(f"\n{'='*60}\nProcessing: {input_file}\n{'='*60}")

    # STAGE 1: Load the raw CSV from Kaggle
    df = pd.read_csv(input_file, on_bad_lines='skip', engine='python')
    print(f"Loaded: {len(df):,}")

    # STAGE 2: Keep only tweets within 7 days before and after Election Day (Nov 3)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at'])
    start = ELECTION_DATE - pd.Timedelta(days=DAYS_BEFORE)
    end   = ELECTION_DATE + pd.Timedelta(days=DAYS_AFTER)
    df = df[(df['created_at'] >= start) & (df['created_at'] <= end)]
    print(f"After time filter ({start.date()} to {end.date()}): {len(df):,}")

    # STAGES 3–4: Remove empty tweets and exact duplicates
    df = df.dropna(subset=['tweet'])
    df = df.drop_duplicates(subset=['tweet'])
    print(f"After empty/dedup: {len(df):,}")

    # STAGES 5–6: Clean tweet text (remove URLs, @mentions, hashtag symbols, HTML),
    # then discard tweets that are too short to carry any sentiment signal
    df['clean_text'] = df['tweet'].apply(clean_text)
    df = df[df['clean_text'].str.len() >= MIN_TEXT_LENGTH]
    print(f"After clean + length filter: {len(df):,}")

    # STAGE 7: Remove spam (promotional phrases, excessive hashtags)
    df = df[~df['tweet'].apply(detect_spam)]
    print(f"After spam removal: {len(df):,}")

    # STAGE 8: Remove likely bot accounts (numeric usernames, zero followers)
    df = df[~df.apply(detect_bot_account, axis=1)]
    print(f"After bot removal: {len(df):,}")

    # STAGES 9–10: Strip emojis and re-check minimum length
    df['clean_text'] = df['clean_text'].apply(remove_emojis)
    for col in ['user_screen_name', 'user_location', 'country', 'state']:
        if col in df.columns:
            df[col] = df[col].apply(remove_emojis)
    df = df[df['clean_text'].str.len() >= MIN_TEXT_LENGTH]
    print(f"After emoji removal: {len(df):,}")

    # STAGE 11: Keep only English-language tweets (no langdetect — custom heuristic)
    df = df[df['clean_text'].apply(is_english)]
    print(f"After English filter: {len(df):,}")

    # STAGE 12: Remove near-duplicate tweets across all users (normalised text match)
    df['norm'] = df['clean_text'].str.lower().str.replace(_NORM_RE, '', regex=True)
    df = df.drop_duplicates(subset=['norm']).drop(columns=['norm'])
    print(f"After global near-dedup: {len(df):,}")

    # STAGE 13: Remove per-user near-duplicates (same user, slightly different opener)
    df = remove_per_user_near_duplicates(df)
    print(f"After per-user near-dedup: {len(df):,}")

    # STAGE 14: Remove low-quality tweets (keyword soup, fragments, hashtag lists)
    df = df[~df['clean_text'].apply(detect_low_quality)]
    print(f"After low-quality removal: {len(df):,}")

    # STAGE 15: Select final columns and save to CSV
    df = df[['created_at', 'clean_text', 'likes', 'retweet_count',
             'user_screen_name', 'user_location', 'country', 'state']]
    df.insert(0, 'id', range(1, len(df) + 1))
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file} ({len(df):,} rows)")


def combine_and_split():
    """Combine Trump+Biden cleaned CSVs, remove cross-duplicates, split before/after election."""
    print(f"\n{'='*60}\nCombining and splitting datasets\n{'='*60}")

    trump_df = pd.read_csv(TRUMP_FINAL)
    biden_df = pd.read_csv(BIDEN_FINAL)
    trump_df['candidate'] = 'trump'
    biden_df['candidate'] = 'biden'
    print(f"Trump: {len(trump_df):,}, Biden: {len(biden_df):,}")

    combined = pd.concat([trump_df, biden_df], ignore_index=True)
    combined['text_norm'] = combined['clean_text'].str.lower().str.replace(_NORM_RE, '', regex=True)
    combined = combined.drop_duplicates(subset=['text_norm'], keep='first').drop(columns=['text_norm'])
    print(f"After cross-duplicate removal: {len(combined):,}")

    combined['clean_text'] = combined['clean_text'].apply(final_clean)
    combined = combined[combined['clean_text'].str.len() >= MIN_TEXT_LENGTH]
    combined['created_at'] = pd.to_datetime(combined['created_at'])
    print(f"After final cleanup: {len(combined):,}")

    election_ts = pd.Timestamp(ELECTION_DATE)
    before = combined[combined['created_at'] < election_ts].sort_values('created_at').reset_index(drop=True)
    after  = combined[combined['created_at'] >= election_ts].sort_values('created_at').reset_index(drop=True)
    before['id'] = range(1, len(before) + 1)
    after['id']  = range(1, len(after)  + 1)

    cols = ['id', 'created_at', 'candidate', 'clean_text', 'likes',
            'retweet_count', 'user_screen_name', 'user_location', 'country', 'state']
    before[cols].to_csv('tweets_before_election.csv', index=False)
    after[cols].to_csv('tweets_after_election.csv', index=False)
    print(f"\ntweets_before_election.csv: {len(before):,} rows")
    print(f"tweets_after_election.csv:  {len(after):,}  rows")

    # Data quality check
    all_tweets = pd.concat([before, after])
    checks = {
        '@mentions':     r'@[A-Za-z]',
        'HTML entities': r'&\w+;|&\d+',
        'URL fragments': r'http|www\.|://|ttps:',
        '# symbols':     r'#',
        'trailing via':  r'\bvia\s*$',
    }
    print("\nData quality check:")
    for label, pattern in checks.items():
        n = all_tweets['clean_text'].str.contains(pattern, regex=True, na=False).sum()
        print(f"  {label}: {'OK' if n == 0 else f'FOUND ({n})'}")


if __name__ == '__main__':
    process_csv('hashtag_donaldtrump.csv', TRUMP_FINAL)
    process_csv('hashtag_joebiden.csv', BIDEN_FINAL)
    combine_and_split()
