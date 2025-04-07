# Fake_News-Detection-Using-Deep-CNN-and-Bi-LSTM
Built a deep learning-based fake news classifier using BiLSTM and attention mechanisms, achieving 98% accuracy on a hybrid dataset. Applied advanced preprocessing, data cleaning, and model evaluation using cross-validation.
# 📰 Fake News Detection Using Deep Learning

This project focuses on detecting fake news articles using a combination of Natural Language Processing (NLP) techniques and deep learning architectures such as BiLSTM and Attention Mechanisms. The model is trained and evaluated on a hybrid dataset containing both real and fake news articles collected from multiple sources.

## 📌 Project Highlights

- Cleaned and merged multiple datasets containing labeled news articles.
- Applied advanced text preprocessing and feature engineering techniques.
- Built a hybrid deep learning model using BiLSTM, CNN, and Multi-Head Attention.
- Evaluated performance using cross-validation and detailed classification metrics.
- Integrated sentiment analysis and readability scores to enhance feature space.

---

## 🧠 Tech Stack

- **Languages & Libraries**: Python, NLTK, Pandas, NumPy, Scikit-learn, TensorFlow (Keras)
- **NLP Tools**: Tokenization, Stopword Removal, TF-IDF, Sentiment Analysis (VADER), Readability Scores
- **Model Architecture**:
  - Embedding Layer
  - Convolutional Layer
  - BiLSTM (Bidirectional Long Short-Term Memory)
  - Multi-Head Attention
  - Dense Layers with Dropout & L2 Regularization

---

## 📁 Dataset

### Sources:
- [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Custom curated datasets (`real1.csv`, `fake1.csv`)

### Structure:
Each entry contains:
- `title`: News headline
- `text`: Full article content
- `label`: 1 for real, 0 for fake

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
Results
Achieved high accuracy and F1-scores across folds

Performance improved with the integration of attention layers and deeper text cleaning

Reduced overfitting using Dropout, BatchNorm, EarlyStopping, and L2 regularization

