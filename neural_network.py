import pandas as pd
from sklearn.model_selection import train_test_split
import collections
import helper
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
# from tensorflow.keras.layers import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Embedding
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

#
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# load the Stanford GloVe model

#glove_input_file = glove_filename
#glove_filename = "glove.6B.100d.txt"
#glove_file = datapath('glove.6B.100d.txt')
#tmp_file = get_tmpfile("test_word2vec.txt")
#_ = glove2word2vec(glove_file, tmp_file)

#model = KeyedVectors.load_word2vec_format(tmp_file)
#print('King: ',model.get_vector('king'))

import tensorflow as tf

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# tests.test_tokenize(tokenize)
# read csv input data
data = pd.read_csv('mixedtranslation.csv')
english_sentences = data['english_sentences'].fillna("")
spanish_sentences = data['spanish_sentences'].fillna("")

english_sentences_aux = helper.load_data('small_vocab_en')
print(english_sentences_aux[0])

X_train, X_test, y_train, y_test = train_test_split(english_sentences, spanish_sentences, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

X_train = X_train.tolist()
X_test = X_test.tolist()
y_train = y_train.tolist()
y_val = y_val.tolist()
X_val = X_val.tolist()
print("-----------------------------------------------------------------------")
print(X_train[0])

# print(data)
#train, test = train_test_split(df, test_size=0.2) 
#X_train, X_test, y_train, y_test = train_test_split( english_sentences, spanish_sentences, test_size=0.2, random_state=1)
def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    x_tk = Tokenizer(char_level=False)
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


# print(english_sentences)
# print("------------")
# print(spanish_sentences)

english_words_counter = collections.Counter([word for sentence in X_train for word in sentence.split()])
spanish_words_counter = collections.Counter([word for sentence in y_train for word in sentence.split()])



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

preproc_english_sentences, preproc_spanish_sentences, english_tokenizer, spanish_tokenizer = preprocess(X_train, y_train)
preproc_english_sentences_val, preproc_spanish_sentences_val, english_tokenizer_val, spanish_tokenizer_val = preprocess(X_val, y_val)
preproc_english_sentences_test, preproc_spanish_sentences_test, english_tokenizer_test, spanish_tokenizer_test = preprocess(X_test, y_test)
print(spanish_tokenizer)
print("----------------------------------------------------------------------")

max_english_sequence_length = preproc_english_sentences.shape[1]
max_spanish_sequence_length = preproc_spanish_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
spanish_vocab_size = len(spanish_tokenizer.word_index)

max_english_sequence_length_val = preproc_english_sentences_val.shape[1]
max_spanish_sequence_length_val = preproc_spanish_sentences_val.shape[1]

""" print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max Spanish sentence length:", max_spanish_sequence_length)
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

""" def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
"""     """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """ """
    # TODO: Build the layers
    learning_rate=0.001
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)
    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


# tests.test_simple_model(simple_model)

# Reshaping the input to work with a basic RNN
# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, max_spanish_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_spanish_sentences.shape[-2], 1))

tmp_x_val = pad(preproc_english_sentences_val, max_spanish_sequence_length_val)
tmp_x_val = tmp_x.reshape((-1, preproc_spanish_sentences_val.shape[-2], 1))

# print((tmp_x[:1])[0])

# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_spanish_sequence_length,
    english_vocab_size,
    spanish_vocab_size)

print(len(tmp_x))
print(len(preproc_spanish_sentences))
#simple_rnn_model.fit(tmp_x, preproc_spanish_sentences, batch_size=1024, epochs=10, validation_data=(tmp_x_val, preproc_spanish_sentences_val))
simple_rnn_model.fit(tmp_x, preproc_spanish_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], spanish_tokenizer))
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
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


def embed_model(input_shape, output_sequence_length, english_vocab_size, spanish_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # Implement
    learning_rate= 0.001
    model = Sequential()
    model.add(Embedding(english_vocab_size, 100, input_length=input_shape[1], input_shape=input_shape[1:]))
     
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(128,  return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(spanish_vocab_size, activation='softmax'))) 
     
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
     
  
    return model


tmp_x = pad(preproc_english_sentences, preproc_spanish_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_spanish_sentences.shape[-2]))

tmp_x_val = pad(preproc_english_sentences_val, preproc_spanish_sentences_val.shape[1])
tmp_x_val = tmp_x_val.reshape((-1, preproc_spanish_sentences.shape[-2]))




embed_rnn_model = embed_model(
    tmp_x.shape,
    preproc_spanish_sentences.shape[1],
    len(english_tokenizer.word_index)+1,
    len(spanish_tokenizer.word_index)+1)

embed_rnn_model.fit(tmp_x, preproc_spanish_sentences, batch_size=300, epochs=10, validation_data=(tmp_x_val, preproc_spanish_sentences_val))
#embed_model.fit(tmp_x, preproc_spanish_sentences, batch_size=1024, epochs=10, validation_split=0.2)

embed_rnn_model.summary()
model_file = "embed_model.h5"
#embed_rnn_model.save(os.path.join(cache_dir, model_file))


print(X_train[:1])
print(y_train[:1])


tmp_x_test = pad(preproc_english_sentences_test, preproc_spanish_sentences_test.shape[1])
tmp_x_test= tmp_x_test.reshape((-1, preproc_spanish_sentences.shape[-2]))

print(logits_to_text(embed_rnn_model.predict(tmp_x_test[:1])[0], spanish_tokenizer))