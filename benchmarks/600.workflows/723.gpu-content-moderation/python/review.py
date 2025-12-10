def calculate_accuracy(posts: list) -> dict:
    true_violations = [p for p in posts if p.get("true_violation") is not None]

    detected_violations = [p for p in posts if p.get("action") in ["REMOVE", "REVIEW"]]

    true_positives = len([
        p for p in posts
        if p.get("true_violation") is not None and p.get("action") in ["REMOVE", "REVIEW"]
    ])

    false_positives = len([
        p for p in posts
        if p.get("true_violation") is None and p.get("action") in ["REMOVE", "REVIEW"]
    ])

    false_negatives = len([
        p for p in posts
        if p.get("true_violation") is not None and p.get("action") not in ["REMOVE", "REVIEW"]
    ])

    precision = true_positives / len(detected_violations) if detected_violations else 0.0
    recall = true_positives / len(true_violations) if true_violations else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_actual_violations": len(true_violations),
    }


def handler(event):
    posts = event.get("posts", [])

    action_counts = {}
    for post in posts:
        action = post.get("action", "UNKNOWN")
        action_counts[action] = action_counts.get(action, 0) + 1

    violation_counts = {}
    for post in posts:
        for violation in post.get("violations", []):
            vtype = violation["type"]
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

    category_stats = {}
    for post in posts:
        category = post.get("category", "unknown")
        toxicity = post.get("toxicity_score", 0)

        if category not in category_stats:
            category_stats[category] = {
                "count": 0,
                "avg_toxicity": 0.0,
                "removed": 0,
            }

        category_stats[category]["count"] += 1
        category_stats[category]["avg_toxicity"] += toxicity
        if post.get("action") == "REMOVE":
            category_stats[category]["removed"] += 1

    for category in category_stats:
        count = category_stats[category]["count"]
        category_stats[category]["avg_toxicity"] = round(
            category_stats[category]["avg_toxicity"] / count, 4
        )

    needs_review = [
        {
            "post_id": p["post_id"],
            "category": p.get("category", "unknown"),
            "toxicity_score": p.get("toxicity_score", 0),
            "spam_score": p.get("spam_score", 0),
            "violations": [v["type"] for v in p.get("violations", [])],
            "sentiment": p.get("sentiment", {}).get("polarity", "neutral"),
        }
        for p in posts
        if p.get("action") == "REVIEW"
    ]

    needs_review.sort(key=lambda p: p["toxicity_score"], reverse=True)

    accuracy = calculate_accuracy(posts)

    avg_toxicity = sum(p.get("toxicity_score", 0) for p in posts) / len(posts)
    avg_spam = sum(p.get("spam_score", 0) for p in posts) / len(posts)

    removal_rate = action_counts.get("REMOVE", 0) / len(posts) if posts else 0
    quality_level = "POOR" if removal_rate > 0.3 else "MODERATE" if removal_rate > 0.15 else "GOOD"

    sentiment_dist = {"positive": 0, "negative": 0, "neutral": 0}
    for post in posts:
        polarity = post.get("sentiment", {}).get("polarity", "neutral")
        sentiment_dist[polarity] = sentiment_dist.get(polarity, 0) + 1

    return {
        "summary": f"Moderated {len(posts)} text posts",
        "total_posts": len(posts),
        "action_summary": action_counts,
        "violation_summary": violation_counts,
        "category_statistics": category_stats,
        "sentiment_distribution": sentiment_dist,
        "quality_level": quality_level,
        "removal_rate": round(removal_rate, 4),
        "avg_toxicity_score": round(avg_toxicity, 4),
        "avg_spam_score": round(avg_spam, 4),
        "posts_needing_review": needs_review[:20],  # Top 20
        "accuracy_metrics": accuracy,
    }
