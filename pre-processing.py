import numpy as np
import string
import num2words

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
# nb_sentences_es = len(spanish_sentences)
# print("nb en sentences : " + str(nb_sentences_en))
# print("nb es sentences : " + str(nb_sentences_es))

indexes = range(nb_sentences_en)
indexes_ten_percent = np.random.choice(indexes, int(nb_sentences_en / 10))
english_sentences = [english_sentences[i] for i in indexes_ten_percent]
spanish_sentences = [spanish_sentences[i] for i in indexes_ten_percent]

"""for i in range(100):
    print(english_sentences[i])
    print(spanish_sentences[i])"""

# pre-processing
english_sentences_preprocessed = []
spanish_sentences_preprocessed = []


def pre_process_list(list_sentences: list[str], list_preprocessed: list[str], language: str) -> list[str]:
    for curr_sentence in list_sentences:
        if curr_sentence != '':
            # lowercase
            curr_sentence = curr_sentence.lower()
            # remove punctuation
            curr_sentence = curr_sentence.translate(str.maketrans('', '', string.punctuation))
            # change numbers into word equivalents
            curr_sentence = ' '.join([num2words.num2words(word, lang=language) if word.isdigit() else word
                                      for word in curr_sentence.split()])
            list_preprocessed.append(curr_sentence)
    return list_preprocessed


english_sentences_preprocessed = pre_process_list(english_sentences, english_sentences_preprocessed, "en")
spanish_sentences_preprocessed = pre_process_list(spanish_sentences, spanish_sentences_preprocessed, "es")

for i in range(100):
    print(english_sentences_preprocessed[i])
    print(spanish_sentences_preprocessed[i])

np.savetxt("pre-processed_data_en.csv", english_sentences_preprocessed, delimiter=" ", fmt='%s')
np.savetxt("pre-processed_data_es.csv", spanish_sentences_preprocessed, delimiter=" ", fmt='%s')