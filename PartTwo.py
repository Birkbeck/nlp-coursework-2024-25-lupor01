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

