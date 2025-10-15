import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Create a custom nltk data folder
nltk_data_dir = "./nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)

# 2. Download resources to that folder
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)  # needed for lemmatizer sometimes

# 3. Tell NLTK where to look for data
nltk.data.path.append(nltk_data_dir)

# 4. Load stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 5. Load dataset
data = pd.read_csv("IMDB Dataset.csv")

# 6. Clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)       # remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove non-letters
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['clean_review'] = data['review'].apply(clean_text)

# 7. Convert to TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_review'])
y = data['sentiment']

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# 10. Evaluate
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
