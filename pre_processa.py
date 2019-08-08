import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import string
import re
from sklearn.utils import shuffle
from nltk.tokenize import TweetTokenizer

punctuation = list(string.punctuation)
stop = stopwords.words('portuguese') + punctuation

regex_str = [
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

regex_url = [
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
]

remove_re = re.compile(r'(' + '|'.join(regex_url) + ')', re.VERBOSE | re.IGNORECASE)
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)

stemmer=nltk.stem.RSLPStemmer()
tknzr = TweetTokenizer()

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(frase, radical,lowercase=True):
    frase = remove_re.sub('',frase.decode('utf-8'))
    tokens = tokenize(frase)
    if radical:
        return [stemmer.stem(token.lower()) for token in tokens]
    else:
        return [token.lower() for token in tokens]

