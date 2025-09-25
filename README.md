---
license: apache-2.0
tags:
- security
- cybersecurity
- wazuh
- transformer
- roberta
- secroberta
- log-analysis
- anomaly-detection
language:
- en
datasets:
- wazuh-assist-dataset
metrics:
- accuracy
- precision
- recall
- f1
library_name: transformers
pipeline_tag: text-classification
---

# Wazuh SecRoBERTa Security Log Classifier

## Model Description

This is a fine-tuned SecRoBERTa model for classifying Wazuh security logs into three categories:
- **Benign (0)**: Normal, safe activities
- **Suspicious (1)**: Potentially concerning activities that require monitoring
- **Malicious (2)**: Confirmed threats requiring immediate action

The model is based on [jackaduma/SecRoBERTa](https://huggingface.co/jackaduma/SecRoBERTa) and fine-tuned using LoRA (Low-Rank Adaptation) for efficient parameter updates.

## Model Architecture

- **Base Model**: SecRoBERTa (Security-focused RoBERTa)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Classification Head**: 3-class classifier
- **Additional Features**: 136-dimensional feature vector for log metadata
- **Max Sequence Length**: 512 tokens

## Training Details

- **Training Framework**: PyTorch + HuggingFace Transformers + PEFT
- **Loss Function**: Focal Loss (for handling class imbalance)
- **Optimization**: AdamW with learning rate scheduling
- **Data**: Wazuh security logs

## Usage

### Using transformers library:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "pyToshka/wazuh-secroberta-full-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare input
text = "Failed login attempt from IP 192.168.1.100"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

# Class mapping
class_names = ["benign", "suspicious", "malicious"]
prediction = class_names[predicted_class]
print(f"Prediction: {prediction}")
```

### Using the project's custom class:

```python
from src.models.secroberta import WazuhSecRoBERTa

# Load model
model = WazuhSecRoBERTa.load_model("pyToshka/wazuh-secroberta-full-v1")

# Make prediction
log_text = "Failed login attempt from IP 192.168.1.100"
prediction, confidence = model.predict(log_text)
print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
```

## Performance

The model achieves strong performance on Wazuh log classification:
- High precision for malicious activity detection
- Good recall for suspicious activity monitoring
- Balanced accuracy across all three classes

## Deployment

This model can be deployed using:
- **ONNX Runtime**: For production inference
- **FastAPI**: REST API server included in the project
- **Docker**: Containerized deployment available

## Citation

```bibtex
@misc{wazuh-assist-2025,
  title={Wazuh SecRoBERTa Security Log Classifier},
  author={Your Organization},
  year={2024},
  howpublished={\url{https://huggingface.co/pyToshka/wazuh-secroberta-full-v1}},
}
```

## License

BSD 3-Clause License


