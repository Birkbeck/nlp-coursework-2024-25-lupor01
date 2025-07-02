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
top_4 = df["party"].value_counts().head(4).index
df = df[df["party"].isin(top_4)]
df = df[df["party"] != "Speaker"]


# a)iii) keeping only rows where 'speech_class' == 'Speech'
df = df[df["speech_class"] == "Speech"]



print(df.shape)
