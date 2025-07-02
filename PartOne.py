#Re-assessment template 2025

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


def read_novels(pathway):
    """ creates a pd.DataFrame as specified in the assignment"""
    ## I started working on the coursework before checking the template on git.
    ## I tried using Path.cwd, but it didn't seem to work for me,
    ## so I kept my original version – it just requires the explicit pathway.
    
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



def nltk_ttr(text):
    """
    Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize.
    """
    tokens = word_tokenize(text)
    tokens = [
        token.lower() for token in tokens
        if token not in punctuation
    ]

    tot_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    return round(unique_tokens/tot_tokens, 3)



def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results



def fk_level(df):
    """
    Print the Flesch-Kincaid scores per novel from a pd.DataFrame.
    The Flesch-Kincaid formula used here is
    FK = .39 * (tot_words / tot_sentences) + 11.8 * (tot_syllables/tot_words) - 15.59
    """
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



def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pkl"):
    """
    Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle (.pkl) file
    """
    df["tokens"] = df["text"].apply(nlp.tokenizer)
    
    store_path.mkdir(parents = True, exist_ok = True) # need to make sure the directory exists!

    with open(store_path/out_name, "wb") as file:
        pickle.dump(df, file)

    return df



def object_counts(df):
    """
    Extracts the most common syntactic objects in parsed documents from pd.DataFrame
    """
 
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
        """
        Helper fuction for subjects_by_verb_count and subjects_by_verb_pmi.
        It normalises the input verb form to a standard form, so that different
        forms of 'hear' are processable (e.g., 'to hear', 'Hear', 'hearing')
        ..useless here, but slightly more robust
        """
        
        if v.startswith("to "):
            v = v[3:]
        doc = nlp(v)
        verb = doc[0]
        return verb.lemma_


def subjects_by_verb_count(df, verb):
    """
    it prints the most common subjects of a given verb in a parsed document from a pd.DataFrame.
    """
    
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
        print(f"{[i for i in frequent_10]}")


def subjects_by_verb_pmi(df, verb):
    """
    it prints the most common subjects of a given verb in a parsed document by PMI
    """
    
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
            pmi[subj] = round(PMI, 4)
        
        top_pmi = sorted(   # from pmi dictionary ––> list of tuples!
            pmi.items(),
            key = lambda item: item[1],
            reverse = True
        )[:10]

        print(f"title: {title}")
        print(f"{[i for i in top_pmi]}")




if __name__ == "__main__":
    """
    insert own pathway ⛔️
    """
    data_folder = "/Users/rick/Desktop/MSc/4 - NLP/0, assessment"
    path_novels = f"{data_folder}/p1-texts/novels"

    print(path_novels)
    
    df = read_novels(path_novels) # this line will fail until you have completed the read_novels function above.
    
    print("\nFirst 5 rows of the dataframe")
    print(df.head(5))
    # nltk.download("cmudict")
    
    print("\nJust about to parse().. returning a df with parsed text")
    df = parse(df)

    print("\nFirst 5 rows of the new df")
    print(df.head(5))

    print("\nMapping each novel to its type-token ratio\n")
    print(get_ttrs(df))
    print("\n\n")
    print(fk_level(df))

    print("\nLoading pickle.pkl file\n")
    df_final = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pkl")
    print("\n.pkl load successfully!\n")
    
    print(f"\nMost common syntactic objects per novel:")
    object_counts(df_final)
    
    print("\nMost common subjects of the verb 'to hear', per novel, by descending frequency:")
    subjects_by_verb_count(df_final, 'hear')

    print("\nMost common subjects of the verb 'to hear', per novel, by descending PMI:")
    subjects_by_verb_pmi(df_final, 'hear')

