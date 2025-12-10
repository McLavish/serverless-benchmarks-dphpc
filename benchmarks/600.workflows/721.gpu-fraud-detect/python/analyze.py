import numpy as np


def simple_neural_network(features: np.ndarray) -> float:
    # Simulated weights for a 2-layer neural network
    n_features = len(features)
    hidden_size = 16

    # Initialize pseudo-random weights based on feature statistics
    seed_val = int(features.sum() * 1000) % 10000
    rng = np.random.default_rng(seed=seed_val)

    # Layer 1: Input -> Hidden
    W1 = rng.normal(0, 0.1, (n_features, hidden_size)).astype(np.float32)
    b1 = rng.normal(0, 0.01, hidden_size).astype(np.float32)
    hidden = np.maximum(0, features @ W1 + b1)  # ReLU activation

    # Layer 2: Hidden -> Output
    W2 = rng.normal(0, 0.1, (hidden_size, 1)).astype(np.float32)
    b2 = rng.normal(0, 0.01, 1).astype(np.float32)
    output = 1.0 / (1.0 + np.exp(-(hidden @ W2 + b2)))  # Sigmoid activation

    return float(output[0])


def rule_based_scoring(transaction: dict) -> float:
    score = 0.0
    features = transaction["features"]

    if features[0] > 0.8:
        score += 0.3

    if features[1] < 0.2 or features[1] > 0.8:
        score += 0.2

    location_dist = np.sqrt(features[2]**2 + features[3]**2)
    if location_dist > 0.7:
        score += 0.25

    if features[4] < 0.15:
        score += 0.15

    if features[5] > 0.7:
        score += 0.1

    return min(score, 1.0)


def detect_patterns(transaction: dict) -> list:
    patterns = []
    features = np.array(transaction["features"])

    if features[0] < 0.1 and features[4] < 0.1:
        patterns.append({
            "type": "card_testing",
            "confidence": 0.85,
        })

    if features[0] > 0.75 and np.sqrt(features[2]**2 + features[3]**2) > 0.6:
        patterns.append({
            "type": "account_takeover",
            "confidence": 0.78,
        })

    if features[4] < 0.05:
        patterns.append({
            "type": "velocity_abuse",
            "confidence": 0.72,
        })

    return patterns


def handler(transaction):
    transaction_id = transaction["transaction_id"]
    features = np.array(transaction["features"], dtype=np.float32)

    ml_score = simple_neural_network(features)

    rule_score = rule_based_scoring(transaction)

    ensemble_score = 0.7 * ml_score + 0.3 * rule_score

    patterns = detect_patterns(transaction)

    is_flagged = ensemble_score > 0.5

    return {
        "transaction_id": transaction_id,
        "ml_score": float(ml_score),
        "rule_score": float(rule_score),
        "ensemble_score": float(ensemble_score),
        "is_flagged": is_flagged,
        "patterns": patterns,
        "true_label": transaction.get("true_label", -1),
    }
