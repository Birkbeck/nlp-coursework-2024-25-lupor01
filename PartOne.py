#Re-assessment template 2025
#just a line for first commit..let's see if it works...
# still trying to see if commiting/pushing works..


# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import os
import glob
from pathlib import Path
import pickle
from collections import Counter

from math import log
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
        
        text, title = row["text"], row["title"]

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
            #else???? # you can probably assume token in cmu..
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

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pkl"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    df["tokens"] = df["text"].apply(nlp.tokenizer)
    
    # need to make sure the directory exists
    store_path.mkdir(parents = True, exist_ok = True)

    with open(store_path/out_name, "wb") as file:
        pickle.dump(df, file)

    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = word_tokenize(text)
    tokens = [
        token.lower() for token in tokens
        if token not in punctuation
    ]

    tot_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    return round(unique_tokens / tot_tokens, 3)


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

def object_counts(df):
    """Extracts the most common syntactic objects in parsed documents from pd.DataFrame"""
    # for _, row in df.iterrows():
    #     text = row["text"]
    #     title = row["title"]
    #     tokens = nlp(text)  # make sure nlp is defined!

    #     counts = {}
    #     for token in tokens:
    #         if token.dep_ == "dobj": 
    #             obj = token.text.lower()
    #             if obj in counts:
    #                 counts[obj] += 1
    #             else:
    #                 counts[obj] = 1
        
    #     most_frequent = sorted(
    #         counts.items(),
    #         key = lambda item: item[1],
    #         reverse = True
    #     )[:10]

    #     print(f"title: {title}")
    #     print(f"most frequent objects (raw counts): {most_frequent}")

    # more streamlined method??
    for _, row in df.iterrows():
        text = row["text"]
        title = row["title"]
        tokens = nlp(text)

        objs = [token.text.lower() for token in tokens if token.dep_ == "dobj"]
        counts = Counter(objs)
        frequent_10 = counts.most_common(10)

        print(f"title: {title}")
        print(f"{[i[0] for i in frequent_10]}")


def standardised_verb(v):
        """ helper fuction for subjects_by_verb_count and subjects_by_verb_pmi"""
        if v.startswith("to "):      # useless here..but slightly more robust
            v = v[3:]
        doc = nlp(v)
        verb = doc[0]
        return verb.lemma_


def subjects_by_verb_count(df, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    
    st_verb = standardised_verb(verb)

    for _, row in df.iterrows():
        text = row["text"]
        title = row["title"]
        tokens = nlp(text)

        target_verbs = [
            token for token in tokens
            if token.lemma_ == st_verb
            and token.pos_ == "VERB"
        ]

        subjects = []
        for token in target_verbs:
            subjects.extend([
                child.text.lower()
                for child in token.children
                if child.dep_ == "nsubj"
            ])

        counts = Counter(subjects)  #careful, needs strings!

        frequent_10 = counts.most_common(10)
        print(f"title: {title}")
        print(f"{[i[0] for i in frequent_10]}")


def subjects_by_verb_pmi(df, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    
    st_verb = standardised_verb(verb)
    
    for _, row in df.iterrows():
        text = row["text"]
        title = row["title"]
        doc = nlp(text)

        text_tokens = [token.text.lower() for token in doc]
        tot_tokens = len(text_tokens)
                                                # Counter needs hashable objects
        doc_counts = Counter(text_tokens)  # can't do Counter(doc), use Counter(tokens.text)
        bigram_counts = Counter()
        verb_count = 0

        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ == st_verb:
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subject = child.text.lower()
                        bigram_counts[subject] += 1

                verb_count += 1
        
        p_verb = verb_count / tot_tokens # doesn't need to be in the bigram_counts loop
        
        pmi = {}
        for subj, count in bigram_counts.items():
            p_bigram = count / tot_tokens
            p_subj = doc_counts[subj] / tot_tokens
            PMI = log((p_bigram) / (p_subj * p_verb))
            pmi[subj] = PMI
        
        top_pmi = sorted(   # from pmi dictionary ––> list of tuples!
            pmi.items(),
            key = lambda item: item[1],
            reverse = True
        )[:10]

        print(f"title: {title}")
        print(f"{[i[0] for i in top_pmi]}")




if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    data_folder = "/Users/rick/Desktop/MSc/4 - NLP/0, assessment"
    path_novels = f"{data_folder}/p1-texts/novels"

    print(path_novels)
    # df = read_novels(path_novels) # this line will fail until you have completed the read_novels function above.
    # print(df.head(5))
    # # nltk.download("cmudict")
    # parse(df)
    # print(df.head())
    # print(get_ttrs(df))
    # print(fk_level(df))
    df_final = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pkl")
    print(df_final.head(3))  # delete when you're done ⛔️
    
    # print(f"\nMost common syntactic objects per novel")
    # object_counts(df_final)
    
    # print("\nMost common subjects of the verb 'to hear', per novel, in descending frequency")
    # subjects_by_verb_count(df_final, 'hear')

    # print(adjective_counts(df_final))
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

