#Re-assessment template 2025
#just a line for first commit..let's see if it works...
# still trying to see if commiting/pushing works..


# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import os
import glob
from pathlib import Path
import pandas as pd
from nltk.corpus import cmudict

from string import punctuation
from nltk import word_tokenize, sent_tokenize
import spacy



nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


def fk_level(df):
    
    cmu = cmudict.dict()

    flesch_dict = {}
    for _, row in df.iterrows():
        text = row["text"]
        title = row["title"]

        sentences = len(sent_tokenize(text))
        tokens = [
            token for token in word_tokenize(text)
            if token not in punctuation
        ]

        syllables = 0
        for token in tokens:
            token = token.lower()
            if token in cmu:
                pronunciations = cmu[token]
                counts = [
                    len([pho for pho in p if pho[-1].isdigit()])
                    for p in pronunciations
                ]
                syllables += sum(counts) / len(counts)
            #else????
        flesch = .39 * (len(tokens)/sentences) + 11.8 * (syllables/len(tokens)) - 15.59

        flesch_dict[title] = round(flesch, 3)

    return flesch_dict

# def count_syl(word, d):
#     """Counts the number of syllables in a word given a dictionary of syllables per word.
#     if the word is not in the dictionary, syllables are estimated by counting vowel clusters

#     Args:
#         word (str): The word to count syllables for.
#         d (dict): A dictionary of syllables per word.

#     Returns:
#         int: The number of syllables in the word.
#     """
#     pass

## I started working on the coursework before seeing the template on git.
## I tried using Path.cwd, but it does not seem to work for me,
## so I stuck to my original version – it just requires the explicit pathway.
def read_novels(pathway): 

    files = glob.glob(os.path.join(pathway, "*.txt"))

    texts, titles, authors, years = [], [], [], []

    for file in files:
        with open(file, "r", encoding = "utf-8") as text:
            texts.append(text.read())
        filename = os.path.basename(file)
        filename = filename.rsplit(".", 1) [0]
        components = filename.split("-")  # title[0], author[1], year[2]

        titles.append(components[0].replace("_", " "))
        authors.append(components[1])
        years.append(components[2])
    
    data = {
        "text": texts,
        "title": titles,
        "author": authors,
        "year": years
    }

    df = pd.DataFrame(data)  # earliest first!!

    return df.sort_values(by = "year", ascending = True) 

# print(read_novels(path_novels).loc[:, ["title", "year"]])

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = word_tokenize(text)
    tokens = [
        token.lower() for token in tokens
        if token not in punctuation
    ]

    tot_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    return unique_tokens / tot_tokens


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


# def get_fks(df):
#     """helper function to add fk scores to a dataframe"""
#     results = {}
#     cmudict = nltk.corpus.cmudict.dict()
#     for i, row in df.iterrows():
#         results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
#     return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass





if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    data_folder = "/Users/rick/Desktop/MSc/4 - NLP/0, assessment"
    path_novels = f"{data_folder}/p1-texts/novels"
    print(path_novels)
    df = read_novels(path_novels) # this line will fail until you have completed the read_novels function above.
    print(df.head(5))
    # nltk.download("cmudict")
    # parse(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(fk_level(df))
    # df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

