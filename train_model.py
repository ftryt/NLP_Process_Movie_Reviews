import pandas as pd
import re
import os
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Reduction of words to their base form
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Text to vectors converters
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Train models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# Load dataset
data = pd.read_csv("IMDB Dataset.csv")

# Create a custom nltk data folder
nltk_data_dir = "./nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)

# Download resources to that folder
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)  # needed for lemmatizer sometimes

# Tell NLTK where to look for data
nltk.data.path.append(nltk_data_dir)

# Load stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()      # lemmatization (accurate)
stemmer = PorterStemmer()             # stemming (very fast)

# Clean text
def clean_text(text):
    """
    This function cleans text from html tags, removes non-letters,
    converts all letters to lower,
    and makes lemmatization (reduce words to their base form: “running” → “run”).
    """

    text = re.sub(r'<.*?>', '', text)       # remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove non-letters
    text = text.lower()
    tokens = text.split()
    # lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # stemming
    # tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Clean out data using lemmatization or stemming
data['clean_review'] = data['review'].apply(clean_text)

# Now we need to convert text to vector

# Bag of Words (BoW) fast approach
# Just creates dict with all words in text and counts them
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(data['clean_review'])

# TF-IDF (Term Frequency – Inverse Document Frequency) accurate approach
# TF-IDF increases the weight of rare, important words,
# and decreases the weight of frequent, uninformative ones (“the”, “movie”, “is”).
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_review'])

# In sentiment two unique values "positive" and "negative"
y = data['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
# Now we train model so it will have parameters studied:
# which words are important;
# which patterns in texts correspond to positive or negative.

# Multinomial Naive Bayes
# How it works (simplified)
# It assumes that all words are independent of each other (naive assumption).
# The model estimates how likely a word is to appear in positive or negative text.
# Then, for new text, it calculates which class has a higher probability.
# Often more accurate than Naive Bayes.
# precision: 85-86% with llemmatization and TF-IDF
# model = MultinomialNB()

# Logistic Regression
# Very popular for binary classification.
# Learns to find the linear boundary between classes.
# Gives good results on text problems.
# precision: 89-90% with lemmatization and TF-IDF
# precision: 88% with lemmatization and Bag of Words
model = LogisticRegression(max_iter=1000)

# SVM (Support Vector Machine)
# A powerful model that searches for the optimal hyperplane separating classes.
# Works well on complex data, but is a bit slower.
# precision: 88% with lemmatization and TF-IDF
# model = LinearSVC()

# Actually train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and TF-IDF translator
os.makedirs("./model/", exist_ok=True)
joblib.dump(model, "./model/sentiment_model.pkl")
joblib.dump(tfidf, "./model/tfidf_vectorizer.pkl")