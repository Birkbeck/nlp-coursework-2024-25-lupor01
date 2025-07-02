from pathlib import Path

import pandas as pd
import nltk
import spacy
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

data_folder = "/Users/rick/Desktop/MSc/4 - NLP/0, assessment"
path_speeches = f"{data_folder}/p2-texts/hansard40000.csv"

df = pd.read_csv(path_speeches)

# print(df.loc[:, ["party", "date", "year"]])

