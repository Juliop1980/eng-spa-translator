import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

filename = "es-en/europarl-v7.es-en.en"
english_sentences = []
english_sentences_character_length = []
with open(filename, 'r', encoding='UTF-8') as file:
    for line in file:
        # print(line.rstrip())
        if line.rstrip() != "":
            english_sentences_character_length.append(len(line.rstrip()))
            english_sentences.append(line.rstrip())

filename = "es-en/europarl-v7.es-en.es"
spanish_sentences = []
spanish_sentences_character_length = []

with open(filename, 'r', encoding='UTF-8') as file:
    for line in file:
        # print(line.rstrip())
        if line.rstrip() != "":
            spanish_sentences_character_length.append(len(line.rstrip()))
            spanish_sentences.append(line.rstrip())


# print(spanish_sentences_character_length)

# average number of characters in word

def list_nb_char_in_words(list_of_sentences: list[str], list_of_lengths) -> list[int]:
    for sentence in list_of_sentences:
        for word in sentence.split():
            list_of_lengths.append(len(word))
    return list_of_lengths


english_word_length = list_nb_char_in_words(english_sentences, [])
spanish_word_length = list_nb_char_in_words(spanish_sentences, [])

print("Average length of English word :" + str(mean(english_word_length)))
print("Average length of Spanish word :" + str(mean(spanish_word_length)))

# total number of sentences
nb_tot_sentences_in_corpus = len(english_sentences) + len(spanish_sentences)
print("There are " + str(nb_tot_sentences_in_corpus) + " sentences in the corpus.")

# average length
avg_length_english = mean(english_sentences_character_length)
print("Average length of an English sentence : " + str(avg_length_english))
avg_length_spanish = mean(spanish_sentences_character_length)
print("Average length of a Spanish sentence : " + str(avg_length_spanish))

list_of_lengths_all = english_sentences_character_length + spanish_sentences_character_length
data = [english_sentences_character_length, spanish_sentences_character_length]

# Making a box plot to show the distribution of lengths
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

# Creating axes instance
bp = ax.boxplot(data, patch_artist=True,
                notch='True', vert=0)

colors = ['#0000FF', '#00FF00']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# changing color and line-width of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#8B008B',
                linewidth=1.5,
                linestyle=":")

# changing color and line-width of
# caps
for cap in bp['caps']:
    cap.set(color='#8B008B',
            linewidth=2)

# changing color and line-width of
# medians
for median in bp['medians']:
    median.set(color='red',
               linewidth=3)

# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker='D',
              color='#e7298a',
              alpha=0.5)

# x-axis labels
ax.set_yticklabels(['English', 'Spanish'])

# Adding title
plt.title("Comparison of length between English and Spanish sentences")

# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks(np.arange(0, max(list_of_lengths_all) + 1, 250))
plt.xlabel("Number of characters in sentences")

# Uncomment next line if you want to save the plot
plt.savefig("insights_of_data/Comparison_length_english_spanish.png")
