import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english')) - {"not"}

def preprocess_text(text: str) -> str:
    """
    Preprocesses text to match the training logic in train_model.py:
    1. Keep only letters (remove numbers, punctuation).
    2. Lowercase.
    3. Remove stopwords (except 'not').
    4. Stem using PorterStemmer.
    """
    # 1. Keep only letters
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    
    # 2. Lowercase
    review = review.lower()
    
    # 3. Split
    review = review.split()
    
    # 4. Stem and remove stopwords
    review = [ps.stem(word) for word in review if word not in stop_words]
    
    # 5. Join
    return ' '.join(review)
