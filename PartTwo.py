from pathlib import Path

import random
import numpy
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import spacy
from nltk.stem import WordNetLemmatizer
from string import punctuation
import contractions

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

data_folder = "/Users/rick/Desktop/MSc/4 - NLP/0, assessment"
path_speeches = f"{data_folder}/p2-texts/hansard40000.csv"

df = pd.read_csv(path_speeches)

# a)i) renaming labour value to 'Labour' 
df["party"] = df["party"].replace({
    "Labour (Co-op)": "Labour"
})


# a)ii) removing rows where value in party col != top 4 parties and removing 'Speaker' value.
# print(df["party"].unique()) 
df = df[df["party"] != "Speaker"] 
top_4 = df["party"].value_counts().head(4).index
df = df[df["party"].isin(top_4)]


# a)iii) keeping only rows where 'speech_class' == 'Speech'
# print(df["speech_class"].unique()) # ok, classes are clean
df = df[df["speech_class"] == "Speech"]


# a)iv) removing rows where text < 1000 characters
df = df[df["speech"].str.len() >= 1000]


print(df.shape)


SEED = 26
# random.seed(SEED) #????
numpy.random.seed(SEED)

stopwords = set(stopwords.words("english"))

# def my_tokenizer(text: str):
#     """
#     - split on whitespaces
#     - remove puntuation
#     - remove stopwords
#     - contractions?!
#     - lemmatize?!
#     """
#     decontracted_text = contractions.fix(text)
#     tokens = word_tokenize(decontracted_text)

#     return [
#         word.lower() for word in tokens
#         if len(word) >2
#         and word not in stopwords
#         and word not in punctuation
#     ]

nlp = spacy.load("en_core_web_sm")

def my_tokenizer(text: str):
    """
    - remove puntuation
    - remove stopwords
    - leave out one-character symbols?
    - smart tokenization with spacy?
    """
    doc = nlp(text)

    return [
        token.text.lower() for token in doc
        if not token.is_digit
        # if not token.like_num  # .is_digit better!
        and not token.is_punct
        and not token.is_stop
        and len(token) >1
    ]
    

vectorizers = [
    # ("Unigrams only", TfidfVectorizer(stop_words = "english", max_features = 3000)),
    # ("Unigrams, bigrams and trigrams", TfidfVectorizer(stop_words = "english", max_features = 3000, ngram_range = (1, 3))),
    ("Adding my tokenizer", TfidfVectorizer(stop_words = "english", max_features = 3000, ngram_range = (1, 3), tokenizer = my_tokenizer))
]

models = [
    ("Random Forest", RandomForestClassifier(n_estimators = 300, random_state = SEED)),
    ("Support Vector Machine", SVC(kernel = "linear"))
]

for vectorizer in vectorizers:
    print(f"\n{vectorizer[0]}".upper())
    features = vectorizer[1].fit_transform(df["speech"])
    labels = df["party"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, stratify = labels, random_state = SEED
    )

    for name, model in models:
        print(f"\n* Classification with {name}:")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        f1 = metrics.f1_score(y_test, pred, average = "macro")
        print(f"\t• Macro F1 score: {round(f1, 3)}")
        # print("\n\t• Classification report:\n")
        # print(metrics.classification_report(y_test, pred, zero_division = 0))