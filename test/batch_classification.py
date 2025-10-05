"""
Batch classification example for multiple logs
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def batch_classify(logs: list, model_path: str = ".."):
    """
    Classify multiple logs in batch for better performance

    Args:
        logs: List of log messages to classify
        model_path: Path to the model directory

    Returns:
        list: List of (label, confidence) tuples
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Tokenize all inputs
    inputs = tokenizer(logs, return_tensors="pt", truncation=True,
                      max_length=512, padding=True)

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values

    # Map to labels
    labels = ["benign", "suspicious", "malicious"]
    results = [
        (labels[pred.item()], conf.item())
        for pred, conf in zip(predicted_classes, confidences)
    ]

    return results


if __name__ == "__main__":
    # Sample logs for batch processing
    logs = [
        "System started successfully",
        "Failed SSH login from 45.33.32.156",
        "User session created for john.doe",
        "Port scan detected from 198.51.100.23",
        "Firewall rule updated",
        "Malware signature detected in file upload",
        "Database backup completed",
        "Privilege escalation attempt blocked",
    ]

    print("Batch Classification Demo\n")
    print("=" * 80)

    results = batch_classify(logs)

    # Display results
    for i, (log, (label, confidence)) in enumerate(zip(logs, results), 1):
        status = "⚠️" if label == "malicious" else "🔍" if label == "suspicious" else "✓"
        print(f"\n{i}. {status} {log}")
        print(f"   → {label.upper()} ({confidence:.2%})")

    # Summary statistics
    print("\n" + "=" * 80)
    print("\nSummary:")
    label_counts = {}
    for label, _ in results:
        label_counts[label] = label_counts.get(label, 0) + 1

    for label, count in label_counts.items():
        print(f"  {label.capitalize()}: {count}")
