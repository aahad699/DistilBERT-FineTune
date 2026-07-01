# 🚀 DistilBERT-FineTune

Fine-tuning **DistilBERT** for **IMDb Movie Review Sentiment Analysis** using the Hugging Face Transformers library. This project also compares a traditional Machine Learning baseline (TF-IDF + Logistic Regression) with a transformer-based approach.

---

## 📌 Project Overview

This project demonstrates how to fine-tune **DistilBERT** for binary text classification (Positive vs Negative sentiment) on the IMDb dataset.

The workflow includes:

- Dataset exploration
- Data visualization
- Baseline ML model
- DistilBERT fine-tuning
- Model evaluation
- Performance comparison
- Saving the trained model
- Sentiment prediction using Hugging Face Pipeline

---

## 📂 Dataset

**Dataset:** IMDb Movie Reviews

- 50,000 movie reviews
- Binary sentiment classification
- 25,000 training samples
- 25,000 testing samples

Loaded directly from Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
```

---

## 🧠 Models

### Baseline Model

- TF-IDF Vectorizer
- Logistic Regression

### Transformer Model

- DistilBERT (`distilbert-base-uncased`)
- Fine-tuned using Hugging Face Trainer API

---

## 📊 Evaluation Metrics

Both models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

## 📈 Project Pipeline

```text
IMDb Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
TF-IDF + Logistic Regression (Baseline)
      │
      ▼
Evaluate Baseline
      │
      ▼
Tokenization
      │
      ▼
Fine-tune DistilBERT
      │
      ▼
Evaluate Model
      │
      ▼
Compare Results
      │
      ▼
Save Model
      │
      ▼
Inference Pipeline
```

---

## 🛠️ Technologies Used

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

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/DistilBERT-FineTune.git

cd DistilBERT-FineTune
```

Install dependencies:

```bash
pip install transformers datasets accelerate scikit-learn matplotlib pandas torch
```

---

## ▶️ Running the Notebook

Open the notebook:

```
DistilBERT_Notebook.ipynb
```

Run all cells in order.

The notebook will:

- Load the dataset
- Train the baseline model
- Fine-tune DistilBERT
- Evaluate performance
- Save the trained model

---

## 💾 Saved Model

After training, the model is saved locally as:

```
sentiment_model/
```

You can reload it using:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="sentiment_model"
)
```

---

## 💬 Example Prediction

```python
classifier(
    "This movie was absolutely amazing!"
)
```

Example output:

```text
[{'label': 'POSITIVE', 'score': 0.998}]
```

---

## 📊 Results

The notebook compares:

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| TF-IDF + Logistic Regression | ✔ | ✔ | ✔ | ✔ |
| DistilBERT (Fine-tuned) | ✔ | ✔ | ✔ | ✔ |

> DistilBERT generally achieves higher performance than the traditional machine learning baseline while leveraging contextual language understanding.

---

## 📁 Repository Structure

```
DistilBERT-FineTune/
│
├── DistilBERT_Notebook.ipynb
├── README.md
└── sentiment_model/        # Generated after training
```

---

## 🎯 Learning Outcomes

This project demonstrates:

- Text preprocessing
- Exploratory Data Analysis (EDA)
- Baseline NLP classification
- Transformer fine-tuning
- Hugging Face Trainer API
- Model evaluation
- Model saving & inference

---

## 🚀 Future Improvements

- Hyperparameter tuning
- Early stopping
- Mixed precision training
- Experiment tracking (Weights & Biases)
- Deploy as a REST API using FastAPI
- Docker containerization
- Streamlit or Gradio interface
- Hugging Face Hub deployment

---

## 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository, open issues, or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!
