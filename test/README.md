# Test Examples

Sample Python scripts demonstrating how to use the Wazuh SecRoBERTa model.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Examples

### 1. Basic Usage (`basic_usage.py`)
Simple single-log classification example with test cases.

```bash
python basic_usage.py
```

### 2. Batch Classification (`batch_classification.py`)
Process multiple logs efficiently in batches.

```bash
python batch_classification.py
```

### 3. HuggingFace Hub Usage (`huggingface_hub_usage.py`)
Load and use the model directly from HuggingFace Hub.

```bash
python huggingface_hub_usage.py
```

## Output

Each script will classify security logs into three categories:
- **Benign** (0): Normal, safe activities
- **Suspicious** (1): Potentially concerning activities
- **Malicious** (2): Confirmed threats

Results include confidence scores for each prediction.
