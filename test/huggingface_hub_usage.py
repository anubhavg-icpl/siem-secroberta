"""
Example using the model from HuggingFace Hub
"""
from transformers import pipeline


def classify_from_hub(log_text: str):
    """
    Classify using model from HuggingFace Hub

    Args:
        log_text: The log message to classify

    Returns:
        dict: Classification result with label and score
    """
    # Load pipeline from hub
    classifier = pipeline(
        "text-classification",
        model="mranv/siem-secroberta-full-v1"
    )

    # Classify
    result = classifier(log_text)[0]

    # Map labels
    label_map = {
        "LABEL_0": "benign",
        "LABEL_1": "suspicious",
        "LABEL_2": "malicious"
    }

    result['label'] = label_map.get(result['label'], result['label'])

    return result


if __name__ == "__main__":
    print("HuggingFace Hub Usage Example\n")
    print("=" * 70)
    print("Note: This will download the model from HuggingFace Hub\n")

    # Test logs
    test_logs = [
        "User logged out normally",
        "Repeated failed login attempts from unknown IP",
        "Ransomware behavior detected",
    ]

    for log in test_logs:
        result = classify_from_hub(log)
        print(f"\nLog: {log}")
        print(f"Result: {result['label'].upper()} (score: {result['score']:.4f})")

    print("\n" + "=" * 70)
