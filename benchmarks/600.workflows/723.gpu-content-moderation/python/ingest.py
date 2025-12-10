import uuid
import numpy as np


POST_CATEGORIES = [
    "news",
    "personal",
    "commercial",
    "entertainment",
    "educational",
    "political",
    "social",
]
LANGUAGES = ["en", "es", "fr", "de", "pt", "it"]

SAFE_WORDS = [
    "cat",
    "dog",
    "food",
    "travel",
    "music",
    "art",
    "game",
    "tech",
    "book",
    "movie",
    "friend",
    "family",
    "work",
    "love",
    "happy",
    "great",
    "amazing",
    "beautiful",
    "fun",
    "enjoy",
    "today",
    "yesterday",
    "tomorrow",
    "share",
    "post",
    "comment",
    "like",
    "follow",
    "update",
]

BORDERLINE_WORDS = [
    "debate",
    "opinion",
    "protest",
    "controversial",
    "argument",
    "disagree",
    "conflict",
    "political",
    "election",
    "policy",
    "criticism",
    "oppose",
    "challenge",
]

UNSAFE_WORDS = [
    "hate",
    "violence",
    "spam",
    "scam",
    "fake",
    "attack",
    "threat",
    "abuse",
    "harass",
    "derogatory",
    "discriminate",
    "offensive",
    "vulgar",
    "explicit",
]


def generate_text_content(avg_tokens: int, seed: int, violation_prob: float = 0.2) -> tuple:
    rng = np.random.default_rng(seed=seed)

    has_violation = rng.random() < violation_prob

    n_tokens = max(10, int(rng.normal(avg_tokens, avg_tokens * 0.3)))

    if has_violation:
        violation_type = rng.choice(
            ["hate_speech", "violence", "spam", "misinformation", "harassment"],
            p=[0.25, 0.2, 0.3, 0.15, 0.1],
        )

        unsafe_ratio = rng.uniform(0.15, 0.4)
        n_unsafe = int(n_tokens * unsafe_ratio)
        n_safe = n_tokens - n_unsafe

        words = list(rng.choice(UNSAFE_WORDS, size=n_unsafe, replace=True)) + list(
            rng.choice(SAFE_WORDS, size=n_safe, replace=True)
        )
    else:
        borderline_ratio = rng.uniform(0, 0.2)
        n_borderline = int(n_tokens * borderline_ratio)
        n_safe = n_tokens - n_borderline

        words = list(rng.choice(BORDERLINE_WORDS, size=n_borderline, replace=True)) + list(
            rng.choice(SAFE_WORDS, size=n_safe, replace=True)
        )
        violation_type = None

    rng.shuffle(words)
    text = " ".join(words)

    return text, violation_type, n_tokens


def handler(event):
    n_posts = int(event["n_posts"])
    avg_tokens = int(event["avg_tokens"])
    batch_id = event.get("batch_id", str(uuid.uuid4())[:8])

    posts = []
    for idx in range(n_posts):
        rng = np.random.default_rng(seed=idx)

        category = rng.choice(POST_CATEGORIES)
        language = rng.choice(LANGUAGES)

        text, violation_type, n_tokens = generate_text_content(avg_tokens, seed=idx)

        post = {
            "post_id": f"post-{batch_id}-{idx:04d}",
            "category": category,
            "language": language,
            "text": text,
            "text_tokens": n_tokens,
            "timestamp": 1700000000 + idx * 60,
            "user_id": f"user_{rng.integers(1000, 9999)}",
            "engagement_score": float(rng.beta(2, 5)),  # Simulates likes/shares
            "batch_id": batch_id,
            "true_violation": violation_type,  # Ground truth for evaluation
        }

        posts.append(post)

    return {"posts": posts}
