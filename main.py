import joblib

# IMPORTANT
# This code is for example of usage and expects for trained model and TF-IDF text to vector approach

# Load model TF-IDF translator
model = joblib.load("./model/sentiment_model.pkl")
tfidf = joblib.load("./model/tfidf_vectorizer.pkl")

# Predict
text = ["This movie was great!", "This movie was so-so."]
x = tfidf.transform(text)
print(model.predict(x))
