import pandas as pd
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'Restaurant_Reviews.tsv')
df = pd.read_csv(dataset_path, delimiter='\t', quoting=3)

# Preprocessing
corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# Vectorization
cv = TfidfVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Retrain on FULL dataset for production
classifier_full = BernoulliNB()
classifier_full.fit(X, y)

print(classifier_full.predict(cv.transform(["The food very bad."])))

# Save model and vectorizer
with open(os.path.join(BASE_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(classifier_full, f)

with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(cv, f)

print("Model and vectorizer saved successfully.")