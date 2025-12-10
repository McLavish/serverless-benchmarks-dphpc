def handler(event):
    slices = event.get("slices", [])

    if not slices:
        return {
            "scan_summary": "No slices to analyze",
            "total_slices": 0,
            "total_anomalies": 0,
        }

    # Aggregate organ statistics across all slices
    total_anomalies = sum(s.get("anomaly_count", 0) for s in slices)

    # Collect all anomalies
    all_anomalies = []
    for slice_result in slices:
        for anomaly in slice_result.get("anomalies", []):
            all_anomalies.append({
                **anomaly,
                "slice_id": slice_result["slice_id"],
                "slice_idx": slice_result["slice_idx"],
            })

    # Sort anomalies by severity
    all_anomalies.sort(key=lambda a: a["severity"], reverse=True)

    # Aggregate organ stats across slices
    organ_aggregate = {}
    for slice_result in slices:
        organ_stats = slice_result.get("organ_stats", {})
        for organ_name, stats in organ_stats.items():
            if organ_name not in organ_aggregate:
                organ_aggregate[organ_name] = {
                    "total_pixels": 0,
                    "avg_percentage": 0.0,
                }
            organ_aggregate[organ_name]["total_pixels"] += stats["pixel_count"]
            organ_aggregate[organ_name]["avg_percentage"] += stats["percentage"]

    # Calculate averages
    n_slices = len(slices)
    for organ_name in organ_aggregate:
        organ_aggregate[organ_name]["avg_percentage"] /= n_slices

    # Generate diagnostic report
    risk_level = "HIGH" if total_anomalies > n_slices * 0.3 else "MEDIUM" if total_anomalies > 0 else "LOW"

    return {
        "scan_summary": f"Analyzed {n_slices} slices",
        "total_slices": n_slices,
        "total_anomalies": total_anomalies,
        "risk_level": risk_level,
        "organ_summary": organ_aggregate,
        "top_anomalies": all_anomalies[:5],  # Top 5 most severe
    }
