from typing import List, Dict


def calculate_severity(
    detections: List[Dict],
    image_width: int,
    image_height: int
) -> Dict:

    if not detections or image_width <= 0 or image_height <= 0:
        return {
            "level": "LOW",
            "coverage_percent": 0.0,
            "cluster_count": 0,
            "reason": "No waste detection above threshold."
        }

    frame_area = image_width * image_height

    total_detection_area = 0.0

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        total_detection_area += width * height

    coverage_percent = (
        total_detection_area / frame_area
    ) * 100

    cluster_count = len(detections)

    # Initial transparent rules.
    # These are configurable starting thresholds,
    # not ground-truth severity standards.
    if coverage_percent > 25 or cluster_count >= 5:
        level = "HIGH"
        reason = (
            f"Waste coverage is {coverage_percent:.2f}% "
            f"with {cluster_count} detected cluster(s)."
        )

    elif coverage_percent > 10 or cluster_count >= 3:
        level = "MEDIUM"
        reason = (
            f"Waste coverage is {coverage_percent:.2f}% "
            f"with {cluster_count} detected cluster(s)."
        )

    else:
        level = "LOW"
        reason = (
            f"Waste coverage is {coverage_percent:.2f}% "
            f"with {cluster_count} detected cluster(s)."
        )

    return {
        "level": level,
        "coverage_percent": round(coverage_percent, 2),
        "cluster_count": cluster_count,
        "reason": reason
    }


def recommended_action(severity: str) -> str:

    actions = {
        "HIGH": "Dispatch cleanup crew",
        "MEDIUM": "Assign to sanitation team",
        "LOW": "Log for next collection cycle"
    }

    return actions.get(
        severity,
        "Review incident"
    )