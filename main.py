import nltk
nltk.download('stopwords')
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.collocations import BigramCollocationFinder

# Load Dataset
df = pd.read_csv("train.csv")
stop_words = set(stopwords.words("english"))
stemmer = LancasterStemmer()

def preprocess(text):
    text = text.lower()
    words = wordpunct_tokenize(text)
    clean_words = []
    for word in words:
        if word.isalpha():
            if word not in stop_words:
                clean_words.append(stemmer.stem(word))
    return " ".join(clean_words)
df["clean_sms"] = df["sms"].apply(preprocess)
print("\nPreprocessed Data")
print(df[["sms","clean_sms"]].head())

# Bigram Collocations

spam_words = []
for msg in df[df["label"] == 1]["clean_sms"]:
    spam_words.extend(msg.split())
finder = BigramCollocationFinder.from_words(spam_words)
print("\nTop Spam Bigrams")
for bigram, freq in finder.ngram_fd.most_common(15):
    print(bigram, "-", freq)

# TF-IDF Vectorization
print()
print("TFIDF Vectorizer")
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["clean_sms"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model= MultinomialNB()
model.fit(X_train, y_train)
prediction= model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, prediction))
#This gives a table that includes columns like precision, recall, f1 score, support
print("\nClassification Report")
print(classification_report(y_test, prediction))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

# SMS Prediction
print()
print("SMS Prediction")

message= input("Enter an SMS: ")
clean_message= preprocess(message)
vector= tfidf.transform([clean_message])
result= model.predict(vector)

if result[0]== 1:
    print("\nPrediction: The SMS is Spam")
else:
    print("\nPrediction: The SMS is Ham")
