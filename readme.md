# DistilBERT Fine-Tuning for Sentiment Analysis

A Natural Language Processing (NLP) project demonstrating fine-tuning of DistilBERT for binary sentiment classification on the IMDb Movie Reviews dataset using the Hugging Face Transformers library.

## Overview

This project compares a traditional machine learning approach with a transformer-based model for sentiment analysis.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Text preprocessing
- Baseline model using TF-IDF and Logistic Regression
- Fine-tuning DistilBERT
- Model evaluation and comparison
- Saving the trained model
- Inference using the Hugging Face Pipeline

---

## Dataset

The project uses the IMDb Movie Reviews dataset from the Hugging Face Datasets library.

- 50,000 movie reviews
- Binary sentiment classification
- Balanced positive and negative samples

Dataset loading:

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
```

---

## Models

### Baseline

- TF-IDF Vectorizer
- Logistic Regression

### Transformer

- DistilBERT (`distilbert-base-uncased`)
- Fine-tuned using the Hugging Face Trainer API

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/DistilBERT-FineTune.git
cd DistilBERT-FineTune
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install transformers datasets accelerate torch scikit-learn pandas matplotlib
```

---

## Project Structure

```
DistilBERT-FineTune/
│
├── DistilBERT_Notebook.ipynb
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Running the Project

Open the notebook:

```
DistilBERT_Notebook.ipynb
```

Run all cells sequentially.

The notebook performs:

1. Dataset loading
2. Exploratory Data Analysis
3. Baseline model training
4. DistilBERT fine-tuning
5. Model evaluation
6. Saving the trained model
7. Sample predictions

---

## Model Evaluation

The following metrics are used to evaluate both models:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

The notebook compares the performance of the baseline model against the fine-tuned DistilBERT model.

---

## Saving the Model

After training, the model is saved locally:

```
sentiment_model/
```

Reload the model using:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="sentiment_model"
)
```

Example:

```python
classifier("This movie was absolutely fantastic.")
```

Output:

```text
[{'label': 'POSITIVE', 'score': 0.998}]
```

---

## Future Improvements

Potential enhancements include:

- Hyperparameter tuning
- Early stopping
- Mixed precision training
- Experiment tracking with Weights & Biases
- FastAPI deployment
- Docker containerization
- Streamlit or Gradio interface
- Deployment on Hugging Face Hub

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Hugging Face Transformers
- Hugging Face Datasets
- Stanford AI Lab (IMDb Dataset)
