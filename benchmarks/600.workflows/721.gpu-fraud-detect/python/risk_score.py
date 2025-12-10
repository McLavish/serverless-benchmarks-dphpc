def calculate_metrics(results: list) -> dict:
    true_positives = sum(1 for r in results if r["is_flagged"] and r.get("true_label") == 1)
    false_positives = sum(1 for r in results if r["is_flagged"] and r.get("true_label") == 0)
    false_negatives = sum(1 for r in results if not r["is_flagged"] and r.get("true_label") == 1)
    true_negatives = sum(1 for r in results if not r["is_flagged"] and r.get("true_label") == 0)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def handler(event):
    transactions = event.get("transactions", [])

    # Aggregate statistics
    total_transactions = len(transactions)
    flagged_transactions = [t for t in transactions if t.get("is_flagged", False)]
    flagged_count = len(flagged_transactions)

    # Calculate average scores
    avg_ml_score = sum(t.get("ml_score", 0) for t in transactions) / total_transactions
    avg_ensemble_score = sum(t.get("ensemble_score", 0) for t in transactions) / total_transactions

    # Collect all detected patterns
    pattern_counts = {}
    for t in transactions:
        for pattern in t.get("patterns", []):
            pattern_type = pattern["type"]
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1

    # Sort flagged transactions by score
    flagged_transactions.sort(key=lambda t: t.get("ensemble_score", 0), reverse=True)

    # Calculate performance metrics
    metrics = calculate_metrics(transactions)

    # Determine overall risk level
    fraud_rate = flagged_count / total_transactions if total_transactions > 0 else 0
    risk_level = "CRITICAL" if fraud_rate > 0.3 else "HIGH" if fraud_rate > 0.15 else "MEDIUM" if fraud_rate > 0.05 else "LOW"

    return {
        "summary": f"Analyzed {total_transactions} transactions",
        "total_transactions": total_transactions,
        "flagged_count": flagged_count,
        "fraud_rate": round(fraud_rate, 4),
        "risk_level": risk_level,
        "avg_ml_score": round(avg_ml_score, 4),
        "avg_ensemble_score": round(avg_ensemble_score, 4),
        "pattern_counts": pattern_counts,
        "top_suspicious": [
            {
                "transaction_id": t["transaction_id"],
                "score": t["ensemble_score"],
                "patterns": [p["type"] for p in t.get("patterns", [])],
            }
            for t in flagged_transactions[:10]
        ],
        "performance_metrics": metrics,
    }
