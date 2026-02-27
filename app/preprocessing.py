"""
Text preprocessing module.
Shared by both the training pipeline and the inference API.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK data is available
for resource in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(resource, quiet=True)

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove punctuation
      4. Remove digits
      5. Remove stopwords
      6. Lemmatize
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)          # URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = re.sub(r"\d+", "", text)                        # digits
    tokens = text.split()
    tokens = [_lemmatizer.lemmatize(w) for w in tokens if w not in _stop_words]
    return " ".join(tokens)
