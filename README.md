# Test Scenario Generator - AI-Powered Model Training

Automated test scenario and test case generation from user stories using fine-tuned Google FLAN-T5 transformer models. This repository contains the complete pipeline for preparing datasets, training, evaluating, and performing inference on an AI model that generates comprehensive test scenarios from natural language user story requirements.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

Generating high-quality test scenarios manually is time-consuming and error-prone. This project automates the process by leveraging state-of-the-art transformer models to generate comprehensive, diverse, and context-aware test scenarios from user story descriptions.

### Key Features

- **Automated Test Scenario Generation** - Convert user stories to detailed test scenarios with a single API call
- **Fine-tuned FLAN-T5 Model** - Pre-trained on curated test generation datasets
- **Full ML Pipeline** - Complete workflow from data preparation to production inference
- **Domain Coverage** - Trained on e-commerce, authentication, forms, and multi-domain user stories
- **Evaluation Metrics** - BLEU, ROUGE, and manual quality assessments included
- **Low Resource Requirements** - Works on CPU; can run on modest hardware
- **HuggingFace Integration** - Leverages HuggingFace datasets and transformers libraries

## Model Details

| Property | Value |
|----------|-------|
| **Model** | google/flan-t5-base |
| **Parameters** | ~250M |
| **Input Max Length** | 256 tokens |
| **Output Max Length** | 512 tokens |
| **Training Method** | Full fine-tuning with Seq2Seq Trainer |
| **Hardware** | CPU/GPU compatible |
| **Framework** | PyTorch + HuggingFace Transformers |

## Project Structure

```
├── 1_prepare_dataset.py           # Dataset preparation & augmentation
├── 2_train_flan_t5.py             # Model fine-tuning script
├── 3_evaluate.py                  # Model evaluation metrics
├── 4_inference.py                 # Inference/generation script
├── README.md                      # This file
│
├── data/
│   ├── train.json                 # Training dataset
│   ├── val.json                   # Validation dataset
│   ├── test.json                  # Test dataset
│   └── test_scenario_dataset/     # HuggingFace dataset format
│       ├── train/                 # Processed training data
│       ├── validation/            # Processed validation data
│       └── test/                  # Processed test data
│
└── models/
    ├── flan_t5_test_gen/          # Final trained model
    │   ├── config.json
    │   ├── model.safetensors      # Model weights
    │   ├── tokenizer.json
    │   ├── generation_config.json
    │   └── all_results.json       # Training metrics
    │
    ├── checkpoint-3/              # Training checkpoints
    ├── checkpoint-12/
    └── evaluation_report.json     # Evaluation results
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda
- 4GB+ RAM (8GB+ recommended)
- Optional: CUDA 11.8+ for GPU acceleration

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/test-scenario-generator.git
   cd test-scenario-generator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies:
   - `transformers>=4.30.0`
   - `datasets>=2.12.0`
   - `torch>=2.0.0`
   - `evaluate>=0.4.0`
   - `rouge-score>=0.1.2`
   - `nltk>=3.8.0`

## Quick Start

### 1. Prepare Dataset

```bash
python 1_prepare_dataset.py
```

This script:
- Loads seed user story data
- Applies data augmentation via templates
- Splits into train/val/test (80/10/10)
- Exports to HuggingFace datasets format
- Saves JSON files to `data/` directory

**Output:** 
- `data/train.json`, `data/val.json`, `data/test.json`
- `data/test_scenario_dataset/` (HuggingFace format)

### 2. Train Model

```bash
python 2_train_flan_t5.py
```

This script:
- Loads pre-trained google/flan-t5-base model
- Fine-tunes on your prepared dataset
- Saves checkpoints during training
- Tracks loss and validation metrics
- Implements early stopping to prevent overfitting

**Output:**
- `models/flan_t5_test_gen/` (final model)
- `models/checkpoint-*/` (intermediate checkpoints)
- Training logs and metrics

**Training Parameters** (configurable in script):
- Batch size: 4
- Learning rate: 3e-4
- Epochs: 10
- Max input length: 256 tokens
- Max target length: 512 tokens

### 3. Evaluate Model

```bash
python 4_evaluate.py
```

This script:
- Loads the trained model
- Evaluates on test dataset
- Computes BLEU and ROUGE scores
- Generates sample predictions
- Saves evaluation report to `models/evaluation_report.json`

**Metrics Included:**
- BLEU Score (precision of n-grams)
- ROUGE-1, ROUGE-2, ROUGE-L (recall-oriented metrics)
- Sequence accuracy
- Sample generations

### 4. Run Inference

```bash
python 5_inference.py
```

This script:
- Loads the trained model and tokenizer
- Provides an interactive interface for test scenario generation
- Takes user stories as input
- Generates comprehensive test scenarios

**Example Input:**
```
User Story: As a registered user, I want to log in with my email and password so that I can access my account dashboard.
```

**Example Output:**
```
Generated Test Scenarios:
1. Valid login with correct email and password → redirects to dashboard
2. Login with incorrect password → show error 'Invalid credentials'
3. Login with unregistered email → show error 'Account not found'
4. Login with empty email field → show validation error
5. Login with SQL injection in email field → system rejects, no crash
...
```

## Model Performance

Evaluation results on test dataset:

| Metric | Score |
|--------|-------|
| BLEU Score | ~0.42 |
| ROUGE-1 | ~0.55 |
| ROUGE-2 | ~0.38 |
| ROUGE-L | ~0.52 |

*Metrics are approximate and may vary based on dataset configuration.*

## Advanced Usage

### Training on Custom Data

Modify the `RAW_DATA` section in `1_prepare_dataset.py`:

```python
RAW_DATA = [
    {
        "user_story": "Your user story here...",
        "test_scenarios": "Test scenarios here...",
    },
    # Add more examples
]
```

### Adjusting Training Parameters

Edit configuration in `2_train_flan_t5.py`:

```python
MODEL_NAME = "google/flan-t5-large"  # Use larger model
BATCH_SIZE = 2                        # Reduce if out of memory
NUM_EPOCHS = 20                       # Train longer
LEARNING_RATE = 5e-4                 # Adjust learning rate
```

### GPU Acceleration

The script automatically detects GPU availability. To force GPU usage:

```bash
export CUDA_VISIBLE_DEVICES=0  # Linux/macOS
set CUDA_VISIBLE_DEVICES=0     # Windows
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Out of Memory (OOM)** | Reduce `BATCH_SIZE` to 2 or 1 in training script |
| **Slow Training on CPU** | Use smaller model variant or train on GPU |
| **Poor Generation Quality** | Increase training epochs or add more seed data |
| **Model Not Found** | Run `1_prepare_dataset.py` before training |
| **CUDA/GPU Issues** | Ensure PyTorch and CUDA versions are compatible |

## Data Format

### Input Format (User Story)
Plain text description of a user requirement:
```
"As a user, I want to [action] so that [benefit]"
```

### Output Format (Test Scenarios)
Numbered list of test cases with expected outcomes:
```
1. Test case description → expected outcome
2. Test case description → expected outcome
...
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Additional seed data for different domains
- Data augmentation strategies
- Alternative model architectures
- Evaluation metrics improvements
- Documentation and examples
- Bug fixes and optimizations

## Performance Tips

- **Smaller model first:** Start with `flan-t5-base` before scaling to `flan-t5-large`
- **Data quality matters:** More curated examples beat more random examples
- **Batch processing:** Use inference script for bulk processing of user stories
- **Caching:** Dataset processing is cached; re-run `1_prepare_dataset.py` to refresh
- **Model versioning:** Save different model versions as you iterate

## Future Enhancements

- [ ] Support for multi-language test scenario generation
- [ ] Fine-tuned models for specific domains (e-commerce, finance, etc.)
- [ ] REST API endpoint for model serving
- [ ] Web UI for interactive scenario generation
- [ ] Integration with popular test automation frameworks (Selenium, Cypress)
- [ ] Prompt engineering for larger models (LLaMA, Mistral)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Google FLAN-T5**: Pre-trained model from Google Research
- **HuggingFace**: Transformers and Datasets libraries
- **PyTorch**: Deep learning framework
- Community contributors and testers

## Citation

If you use this project in your research, please cite:

```bibtex
@software{test_scenario_generator_2026,
  author = Shyamchandar,
  title = {Test Scenario Generator: AI-Powered Test Case Generation},
  year = 2026,
  url = {https://github.com/yourusername/test-scenario-generator}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: your.email@example.com
- Documentation: See individual script comments


---

**Note:** This project is under active development. Feedback and contributions are highly appreciated!
