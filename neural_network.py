import pandas as pd
from sklearn.model_selection import train_test_split
import collections
#import helper
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from tensorflow.keras.layers import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
#from nltk.tokenize import tokenize



def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    x_tk = Tokenizer()
    x_tk.fit_on_texts(x)
 
    return x_tk.texts_to_sequences(x), x_tk


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post', truncating='post')



#tests.test_tokenize(tokenize)
# read csv input data
data = pd.read_csv('mixedtranslation.csv')
english_sentences = data['english_sentences'].fillna("")
spanish_sentences = data['spanish_sentences'].fillna("")
#print(data)
#train, test = train_test_split(df, test_size=0.2)
#X_train, X_test, y_train, y_test = train_test_split(english_sentences, spanish_sentences, test_size=0.2, random_state=1)



#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

#print(english_sentences)
#print("------------")
#print(spanish_sentences)

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
spanish_words_counter = collections.Counter([word for sentence in spanish_sentences for word in sentence.split()])
"""  
print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in spanish_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(spanish_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*spanish_words_counter.most_common(10)))[0]) + '"')
 """
def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
 
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
 
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
 
    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_spanish_sentences, english_tokenizer, spanish_tokenizer =preprocess(english_sentences, spanish_sentences)
     
max_english_sequence_length = preproc_english_sentences.shape[1]
max_spanish_sequence_length = preproc_spanish_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
spanish_vocab_size = len(spanish_tokenizer.word_index)
 
""" print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max Spanishsentence length:", max_spanish_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("Spanish vocabulary size:", spanish_vocab_size) """


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
 
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
""" for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent)) """
""" # Drop first column of dataframe
df = df.iloc[:, 1:]
dataTypeDict = dict(df.dtypes)

df["spam"] = df["spam"].astype('category').cat.codes

train, test = train_test_split(df, test_size=0.2)

emails_list = df['text'].tolist()

vectorizer = CountVectorizer(min_df=0, lowercase=False)

vectorizer.fit(emails_list)
emails_list = df['text'].values
y = df['spam'].values

emails_train, emails_test, y_train, y_test = train_test_split(emails_list, y, test_size=0.2, random_state=1000) """