# 🎬 Sentiment Analysis - IMDB Movie Reviews

A simple yet effective **text classification (sentiment analysis)** project built with Python and scikit-learn.  
The model analyzes IMDB movie reviews and predicts whether a review is **positive** or **negative**.

It demonstrates the full NLP pipeline - from **text preprocessing**, **vectorization (TF-IDF / Bag-of-Words)**, and **model training**, to **saving/loading** the trained model for real-time use.

---

## 📦 Installation

Clone this repository and install all dependencies:

```bash
git clone https://github.com/ftryt/NLP_Process_Movie_Reviews
cd NLP_Process_Movie_Reviews
pip install -r requirements.txt
```

> 💡 Make sure you have **Python ≥ 3.8** and `pip` installed.

---

## 🧰 Required Files

Before training:

- Place the **IMDB dataset** (`IMDB Dataset.csv`) in the project root directory.  
  You can download it here:  
  [📁 IMDB Dataset of 50K Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## ⚙️ Training the Model

The file **`train_model.py`** is well-documented and includes explanations about various models and text transformation methods.  
You can easily modify and retrain the model as you wish.

To train and evaluate the sentiment analysis model, simply run:

```bash
python train_model.py
```

This will:

- Load and preprocess the dataset  
- Train the model (default: **Logistic Regression**)  
- Print accuracy, classification report, and confusion matrix  
- Save trained artifacts into `./model/`:
  - `sentiment_model.pkl`
  - `tfidf_vectorizer.pkl`

---

## 🧪 Using the Trained Model

You can use the trained model directly by running:

```bash
python main.py
```

or import it into your own scripts.  
The model and vectorizer are saved in `./model/` and can be used as shown in `main.py`.

Example:

```python
import joblib

model = joblib.load("./model/sentiment_model.pkl")
tfidf = joblib.load("./model/tfidf_vectorizer.pkl")

texts = ["This movie was great!", "This movie was so-so."]
X = tfidf.transform(texts)
print(model.predict(X))
```

Output:
```
['positive' 'negative']
```

---

## 📊 Model Comparison Notes

| Model                       | Accuracy | Speed       | Notes                                    |
|-----------------------------|----------|-------------|------------------------------------------|
| **Multinomial Naive Bayes** | ~85%     | ⚡ Very Fast | Simple, efficient for text               |
| **Logistic Regression**     | ~89–90%  | ⚡ Fast      | Great balance between speed and accuracy |
| **SVM (LinearSVC)**         | ~88–90%  | ⚖️ Moderate | Slightly slower, robust for complex data |

