import uuid
import numpy as np


TRANSACTION_TYPES = ["purchase", "withdrawal", "transfer", "payment", "deposit"]
MERCHANT_CATEGORIES = ["retail", "online", "travel", "entertainment", "utilities", "groceries"]
COUNTRIES = ["US", "UK", "CA", "DE", "FR", "JP", "AU"]


def generate_transaction_features(n_features: int, seed: int, fraud_prob: float = 0.15) -> tuple:
    rng = np.random.default_rng(seed=seed)

    # Determine if this transaction is fraudulent
    is_fraud = rng.random() < fraud_prob

    features = np.zeros(n_features, dtype=np.float32)

    # Feature 0: Transaction amount (normalized)
    if is_fraud:
        # Fraudulent transactions tend to be larger or at unusual amounts
        features[0] = rng.uniform(0.7, 1.0)
    else:
        features[0] = rng.beta(2, 5)  # Most legitimate transactions are smaller

    # Feature 1: Time of day (0-1, where 0 = midnight, 0.5 = noon)
    if is_fraud:
        # Fraud more likely at odd hours
        features[1] = rng.choice([rng.uniform(0.0, 0.2), rng.uniform(0.8, 1.0)])
    else:
        features[1] = rng.beta(2, 2)

    # Feature 2-3: Location distance from home (lat/lon deviation)
    if is_fraud:
        features[2] = rng.uniform(0.5, 1.0)  # Far from home
        features[3] = rng.uniform(0.5, 1.0)
    else:
        features[2] = rng.exponential(0.2)
        features[3] = rng.exponential(0.2)

    # Feature 4: Velocity (time since last transaction, normalized)
    if is_fraud:
        features[4] = rng.uniform(0.0, 0.1)  # Quick succession
    else:
        features[4] = rng.uniform(0.2, 1.0)

    # Feature 5: Merchant risk score
    if is_fraud:
        features[5] = rng.uniform(0.6, 1.0)
    else:
        features[5] = rng.uniform(0.0, 0.5)

    # Fill remaining features with correlated noise
    for i in range(6, n_features):
        if is_fraud:
            features[i] = rng.normal(0.6, 0.2)
        else:
            features[i] = rng.normal(0.3, 0.2)

    # Clip to [0, 1]
    features = np.clip(features, 0, 1)

    return features.tolist(), int(is_fraud)


def handler(event):
    n_transactions = int(event["n_transactions"])
    n_features = int(event["n_features"])
    batch_id = event.get("batch_id", str(uuid.uuid4())[:8])

    transactions = []
    for idx in range(n_transactions):
        features, is_fraud = generate_transaction_features(n_features, seed=idx)

        rng = np.random.default_rng(seed=idx)
        transaction = {
            "transaction_id": f"txn-{batch_id}-{idx:04d}",
            "features": features,
            "true_label": is_fraud,  # Ground truth for evaluation
            "amount": float(features[0] * 10000),  # Scale to realistic amount
            "timestamp": 1700000000 + idx * 300,  # Sequential timestamps
            "merchant_category": rng.choice(MERCHANT_CATEGORIES),
            "country": rng.choice(COUNTRIES),
            "batch_id": batch_id,
        }
        transactions.append(transaction)

    return {"transactions": transactions}
