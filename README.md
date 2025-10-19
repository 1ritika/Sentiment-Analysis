# 🧠 Sentiment Analysis — Bag-of-Words & Word2Vec

This project implements **sentiment classification on movie reviews** using both classical ML and embedding-based NLP techniques.  
It compares **Bag-of-Words** and **Word2Vec embeddings**, showing how richer representations improve semantic understanding.

---

## 🚀 Features
- Multinomial **Naive Bayes** classifier (from scratch).  
- **Bag-of-Words Logistic Regression** with tokenization and stop-word removal.  
- **Word2Vec-based Logistic Regression** using averaged pretrained **GoogleNews-300** embeddings.  
- **Analogy & Similarity Tests** (e.g., `king – man + woman ≈ queen`) to validate semantic relationships.  
- **Bias Exploration** (e.g., `man:doctor :: woman:nurse`).  
- Achieved **≈80 % validation accuracy** on a binary movie-review sentiment dataset.

---

## 🧩 Dataset
- Binary sentiment dataset (0 = negative, 1 = positive) derived from movie reviews.  
- Automatically downloaded from the provided Google Sheets URLs when running the script.

---

## 🧮 Models Implemented
| Model | Representation | Type | Key Insight |
|-------|----------------|------|--------------|
| Naive Bayes | Count Vectorizer | Generative | Simple interpretable baseline |
| Logistic Regression | Bag-of-Words | Discriminative | Efficient linear text model |
| Logistic Regression | Word2Vec (300-D) | Embedding-based | Captures semantic similarity |

---

## 📈 Results
| Method | Validation Accuracy |
|---------|--------------------|
| Bag-of-Words (LogReg) | ~0.80 |
| Word2Vec (LogReg) | ~0.82 |

Word2Vec embeddings showed better generalization and closer alignment between **cosine similarity** and **semantic proximity**.

---

## 🛠️ Requirements
```bash
pip install numpy pandas nltk gensim scikit-learn
