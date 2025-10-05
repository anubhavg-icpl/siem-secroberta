"""
Basic usage example for Wazuh SecRoBERTa model
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def classify_log(log_text: str, model_path: str = ".."):
    """
    Classify a security log using the Wazuh SecRoBERTa model

    Args:
        log_text: The log message to classify
        model_path: Path to the model directory (default: parent directory)

    Returns:
        tuple: (label, confidence)
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Tokenize input
    inputs = tokenizer(log_text, return_tensors="pt", truncation=True, max_length=512)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    # Map to label
    labels = ["benign", "suspicious", "malicious"]

    return labels[predicted_class], confidence


if __name__ == "__main__":
    # Test samples
    test_logs = [
        "User admin logged in successfully from 192.168.1.10",
        "Failed login attempt from IP 192.168.1.100",
        "Multiple failed authentication attempts detected from 10.0.0.50",
        "SSH brute force attack detected from 203.0.113.42",
        "Normal system update completed successfully",
        "Unauthorized access attempt to /etc/passwd",
    ]

    print("Wazuh SecRoBERTa - Security Log Classification\n")
    print("=" * 70)

    for log in test_logs:
        label, confidence = classify_log(log)
        print(f"\nLog: {log}")
        print(f"Classification: {label.upper()} (confidence: {confidence:.2%})")

    print("\n" + "=" * 70)
