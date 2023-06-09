import numpy as np
import string
import num2words
import pandas as pd
from typing import List
import csv

MAX_LENGTH_SENTENCES = 10

filename_en = "es-en/europarl-v7.es-en.en"
english_sentences = []
with open(filename_en, 'r', encoding='UTF-8') as file:
    for line in file:
        english_sentences.append(line.rstrip())

filename_es = "es-en/europarl-v7.es-en.es"
spanish_sentences = []
with open(filename_es, 'r', encoding='UTF-8') as file:
    for line in file:
        spanish_sentences.append(line.rstrip())

nb_sentences_en = len(english_sentences)
nb_sentences_es = len(spanish_sentences)
print("nb en sentences : " + str(nb_sentences_en))
print("nb es sentences : " + str(nb_sentences_es))
print("nb tot sentences : " + str(nb_sentences_en + nb_sentences_es))

indexes = range(nb_sentences_en)
# to randomly select 10 percent of the data
# indexes_ten_percent = np.random.choice(indexes, int(nb_sentences_en / 10))
# randomly select 1 percent so the dataset is smaller
indexes_ten_percent = np.random.choice(indexes, int(nb_sentences_en / 100))
english_sentences = [english_sentences[i] for i in indexes_ten_percent]
spanish_sentences = [spanish_sentences[i] for i in indexes_ten_percent]

# pre-processing
english_sentences_preprocessed = []
spanish_sentences_preprocessed = []


def pre_process_list(list_sentences: list[str], list_preprocessed: list[str], language: str) -> list[str]:
    for curr_sentence in list_sentences:
        # lowercase
        curr_sentence = curr_sentence.lower()
        # remove punctuation
        curr_sentence = curr_sentence.translate(str.maketrans('', '', string.punctuation))
        # change numbers into word equivalents and remove white spaces
        curr_sentence = ' '.join([num2words.num2words(word, lang=language) if word.isdigit() else word
                                  for word in curr_sentence.split()])
        list_preprocessed.append(curr_sentence)
    return list_preprocessed


print(len(english_sentences))
print(len(spanish_sentences))
english_sentences_preprocessed = pre_process_list(english_sentences, english_sentences_preprocessed, "en")
spanish_sentences_preprocessed = pre_process_list(spanish_sentences, spanish_sentences_preprocessed, "es")

"""for i in range(10):
    print(english_sentences_preprocessed[i])
    print(spanish_sentences_preprocessed[i])

print(len(english_sentences_preprocessed))
print(len(spanish_sentences_preprocessed))"""
""" 
headerList = ['english_sentences', 'spanish_stences']
with open('mixedtranslation.csv', 'w') as f:
    writer = csv.writer(f,fieldnames = headerList)
    writer.writerows(zip(english_sentences_preprocessed, spanish_sentences_preprocessed)) """

with open('mixedtranslation.csv', 'w') as csvfile:
    fieldnames = ['english_sentences', 'spanish_sentences']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(english_sentences_preprocessed)):
        if len(english_sentences_preprocessed[i].split(" ")) <= MAX_LENGTH_SENTENCES and \
                len(spanish_sentences_preprocessed[i].split(" ")) <= MAX_LENGTH_SENTENCES:
            writer.writerow({'english_sentences': english_sentences_preprocessed[i],
                             'spanish_sentences': spanish_sentences_preprocessed[i]})
