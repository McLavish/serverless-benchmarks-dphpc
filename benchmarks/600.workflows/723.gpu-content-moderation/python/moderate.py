import numpy as np


UNSAFE_KEYWORDS = [
    "hate",
    "violence",
    "spam",
    "scam",
    "fake",
    "attack",
    "threat",
    "abuse",
    "harass",
]
BORDERLINE_KEYWORDS = ["controversial", "debate", "protest", "argument"]


def tokenize_and_embed(text: str, seed: int) -> np.ndarray:
    words = text.lower().split()
    n_tokens = len(words)

    rng = np.random.default_rng(seed=seed)
    embedding_dim = 768

    embeddings = rng.normal(0, 0.1, (n_tokens, embedding_dim)).astype(np.float32)

    for i, word in enumerate(words):
        if any(unsafe in word for unsafe in UNSAFE_KEYWORDS):
            embeddings[i] += rng.normal(2.0, 0.5, embedding_dim).astype(np.float32)
        elif any(border in word for border in BORDERLINE_KEYWORDS):
            embeddings[i] += rng.normal(0.5, 0.3, embedding_dim).astype(np.float32)

    sentence_embedding = embeddings.mean(axis=0)

    return sentence_embedding


def transformer_classification(embedding: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed=seed)

    # Simulate classification head (linear layer + softmax)
    embedding_norm = np.linalg.norm(embedding)

    # Higher norm suggests more extreme content (either very safe or very toxic)
    base_score = 1.0 / (1.0 + np.exp(-embedding_norm / 10))

    noise = rng.normal(0, 0.05)
    toxicity_score = np.clip(base_score + noise, 0, 1)

    return {
        "toxicity_score": float(toxicity_score),
        "embedding_norm": float(embedding_norm),
    }


def classify_violation_type(text: str, toxicity_score: float, seed: int) -> list:
    rng = np.random.default_rng(seed=seed)
    violations = []

    words = text.lower()

    if "hate" in words or "discriminate" in words or "derogatory" in words:
        confidence = min(0.95, toxicity_score + rng.uniform(0.1, 0.2))
        violations.append(
            {
                "type": "hate_speech",
                "confidence": float(confidence),
            }
        )

    if "violence" in words or "attack" in words or "threat" in words:
        confidence = min(0.95, toxicity_score + rng.uniform(0.1, 0.2))
        violations.append(
            {
                "type": "violence",
                "confidence": float(confidence),
            }
        )

    if "spam" in words or "scam" in words:
        confidence = min(0.9, toxicity_score + rng.uniform(0.05, 0.15))
        violations.append(
            {
                "type": "spam",
                "confidence": float(confidence),
            }
        )

    if "fake" in words:
        confidence = min(0.85, toxicity_score + rng.uniform(0.05, 0.15))
        violations.append(
            {
                "type": "misinformation",
                "confidence": float(confidence),
            }
        )

    if "harass" in words or "abuse" in words:
        confidence = min(0.9, toxicity_score + rng.uniform(0.1, 0.2))
        violations.append(
            {
                "type": "harassment",
                "confidence": float(confidence),
            }
        )

    return violations


def sentiment_analysis(text: str, seed: int) -> dict:
    rng = np.random.default_rng(seed=seed)

    positive_words = ["happy", "love", "great", "amazing", "beautiful", "enjoy", "fun"]
    negative_words = ["hate", "violence", "abuse", "offensive", "threat"]

    words = text.lower().split()
    pos_count = sum(1 for w in words if any(p in w for p in positive_words))
    neg_count = sum(1 for w in words if any(n in w for n in negative_words))

    total = len(words)
    if total > 0:
        sentiment_score = (pos_count - neg_count) / total
    else:
        sentiment_score = 0.0

    # Add noise
    sentiment_score += rng.normal(0, 0.1)
    sentiment_score = np.clip(sentiment_score, -1, 1)

    return {
        "sentiment_score": float(sentiment_score),
        "polarity": "positive"
        if sentiment_score > 0.1
        else "negative"
        if sentiment_score < -0.1
        else "neutral",
    }


def calculate_spam_score(post: dict) -> float:
    spam_score = 0.0

    text = post.get("text", "")
    words = text.lower().split()

    unique_words = len(set(words))
    if unique_words < len(words) * 0.3:
        spam_score += 0.4

    if "spam" in text or "scam" in text:
        spam_score += 0.5

    engagement = post.get("engagement_score", 0.5)
    if engagement < 0.1:
        spam_score += 0.2

    return min(spam_score, 1.0)


def handler(post):
    post_id = post["post_id"]
    text = post.get("text", "")
    category = post.get("category", "unknown")

    seed = hash(post_id) % 100000

    embedding = tokenize_and_embed(text, seed)

    toxicity_result = transformer_classification(embedding, seed + 1)
    toxicity_score = toxicity_result["toxicity_score"]

    violations = classify_violation_type(text, toxicity_score, seed + 2)

    sentiment = sentiment_analysis(text, seed + 3)

    spam_score = calculate_spam_score(post)

    overall_score = (
        0.5 * toxicity_score
        + 0.3 * spam_score
        + 0.2 * (1.0 if sentiment["sentiment_score"] < -0.3 else 0.0)
    )

    if overall_score > 0.75:
        action = "REMOVE"
    elif overall_score > 0.5:
        action = "REVIEW"
    elif overall_score > 0.3:
        action = "FLAG"
    else:
        action = "APPROVE"

    if spam_score > 0.7:
        action = "REMOVE"

    return {
        "post_id": post_id,
        "category": category,
        "toxicity_score": float(toxicity_score),
        "spam_score": float(spam_score),
        "sentiment": sentiment,
        "violations": violations,
        "overall_score": float(overall_score),
        "action": action,
        "true_violation": post.get("true_violation"),
    }
