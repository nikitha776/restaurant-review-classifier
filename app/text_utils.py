import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure resources are downloaded (this might need to be handled carefully in production)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english')) - {"not"}

NEGATION_TOKENS = {
    "not", "no", "never", "n't", "none", "nobody", "nothing", "neither",
    "nowhere", "hardly", "scarcely", "barely"
}

def mark_negation_tokens(text: str) -> str:
    """
    Appends 'NOT_' to words following negation tokens until a punctuation mark is encountered.
    """
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
    negated = False
    out = []
    for tok in tokens:
        if tok in {'.', '!', '?', ';', ':'}:
            negated = False
            out.append(tok)
        elif tok in NEGATION_TOKENS:
            negated = True
            out.append(tok)
        else:
            out.append("NOT_" + tok if negated else tok)
    return " ".join(out)

def preprocess_text_with_negation(text: str) -> str:
    """
    Cleans text, handles negations, removes stopwords (except negations), and stems words.
    """
    # Basic cleaning: allow alphanumeric, spaces, and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:\'`-]', ' ', str(text))
    
    # Mark negations
    text = mark_negation_tokens(text)
    
    tokens = []
    for tok in text.split():
        if tok.startswith("NOT_"):
            word = tok[4:]
            # Check if the base word is a stop word? The original notebook logic:
            # if word not in stop_words: tokens.append("NOT_" + ps.stem(word))
            if word not in stop_words:
                tokens.append("NOT_" + ps.stem(word))
        else:
            if tok in NEGATION_TOKENS:
                tokens.append(tok)
            elif tok not in stop_words:
                tokens.append(ps.stem(tok))
    
    return " ".join(tokens)
