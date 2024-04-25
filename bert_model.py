"""
* Running bert for individual sets
* Running bert for whole dataset
* Running word2vec for individual sets
* Running word2vec for whole dataset
"""


import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

! pip install transformers

# Declaring some visualization methods to plot accuracy and model diagram
def plot_accuracy_curve(history):
  import matplotlib.pyplot as plt
  plt.plot(history.history['loss'])
  plt.plot(history.history['mae'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

def plot_acrchitecture(filename, model):
  from keras.utils import plot_model
  plot_model(model, to_file=str(filename) + '.png')

# method to split data into sets
def split_in_sets(data):
    essay_sets = []
    min_scores = []
    max_scores = []
    for s in range(1,9):
        essay_set = data[data["essay_set"] == s]
        essay_set.dropna(axis=1, inplace=True)
        n, d = essay_set.shape
        set_scores = essay_set["domain1_score"]
        print ("Set", s, ": Essays = ", n , "\t Attributes = ", d)
        min_scores.append(set_scores.min())
        max_scores.append(set_scores.max())
        essay_sets.append(essay_set)
    return (essay_sets, min_scores, max_scores)

"""Here, we can see the data we need to operate. We essentially drops the column, we dont need and keep the domain_score only along with essay text."""

dataset_path = "./data/training_set_rel3.tsv"
data = pd.read_csv(dataset_path, sep="\t", encoding="ISO-8859-1")
min_scores = [2, 1, 0, 0, 0, 0, 0, 0]
max_scores = [12, 6, 3, 3, 4, 4, 30, 60]
essay_sets, data_min_scores, data_max_scores = split_in_sets(data)
set1, set2, set3, set4, set5, set6, set7, set8 = tuple(essay_sets)
data.dropna(axis=1, inplace=True)
data.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
set1.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
set2.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
set3.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
set4.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
set5.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
set6.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
set7.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
set8.drop(columns=["rater1_domain1", "rater2_domain1"], inplace=True)
sets = [set1,set2,set3,set4,set5,set6,set7,set8]
data.head()

""" There are named entity tags as you can see above, which can impact our model. so we need to remove them from the dataframe. So we will extend our stopwords set later on. But first we need to create all possible ner tags."""

cap = ['@CAPS'+str(i) for i in range(100)]
loc = ['@LOCATION'+str(i) for i in range(100)]
org =['@ORGANIZATION'+str(i) for i in range(100)]
per = ['@PERSON'+str(i) for i in range(100)]
date = ['@DATE'+str(i) for i in range(100)]
time = ['@TIME'+str(i) for i in range(100)]
money = ['@MONEY'+str(i) for i in range(100)]
ner =  cap + loc + org + per + date + time + money

"""Some utility functions declarations needed to convert the raw essay to word list."""

import collections
top10 = collections.defaultdict(int)
def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    #top10 = collections.defaultdict(int)
    if remove_stopwords:
        stops = stopwords.words("english")
        stops.extend(ner)
        for word in words:
          if word not in stops:
            # words.append(w)
            top10[word]+=1
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

"""Below we declare the model. here the model is of older version. for running the most recent version please refer to readme, which inputs the model_type as well in terms of hyperparameter."""

from keras.layers import Embedding, Input, LSTM, Dense, Dropout, Lambda, Flatten, Bidirectional, Conv2D, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential,Model, load_model, model_from_config
import keras.backend as K

def get_model(Hidden_dim1=400, Hidden_dim2=128, return_sequences = True, dropout=0.5, recurrent_dropout=0.4, input_size=768, activation='relu', bidirectional = False):
    """Define the model."""
    model = Sequential()
    if bidirectional:
        model.add(Bidirectional(LSTM(Hidden_dim1,return_sequences=return_sequences , dropout=0.4, recurrent_dropout=recurrent_dropout), input_shape=[1, input_size]))
        model.add(Bidirectional(LSTM(Hidden_dim2, recurrent_dropout=recurrent_dropout)))
    else:
        model.add(LSTM(Hidden_dim1, dropout=0.4, recurrent_dropout=recurrent_dropout, input_shape=[1, input_size], return_sequences=return_sequences))
        model.add(LSTM(Hidden_dim2, recurrent_dropout=recurrent_dropout))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=activation))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model

def get_model_CNN(Hidden_dim1=400, Hidden_dim2=128, return_sequences = True, dropout=0.5, recurrent_dropout=0.4, input_size=768,output_dims=10380, activation='relu', bidirectional = False):
    """Define the model."""
    inputs = Input(shape=(768,1))
    x = Conv1D(64, 3, strides=1, padding='same', activation='relu')(inputs)
    #Cuts the size of the output in half, maxing over every 2 inputs
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    outputs = Dense(output_dims, activation='relu')(x)
    model = Model(inputs=inputs, outputs=outputs, name='CNN')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','mse'])
    model.summary()
    return model

"""Below we will run the model for all sets using BERT"""

## Sets experiment BERT
import time
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
set_count = 1
all_sets_score = []
for s in sets:
  print("SET {}".format(set_count))
  X = s
  y = s['domain1_score']
  cv = KFold(n_splits=5, shuffle=True)
  cv_data = cv.split(X)
  results = []
  prediction_list = []
  fold_count =1
  cuda = torch.device('cuda')
  # For DistilBERT:
  model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
  ## Want BERT instead of distilBERT? Uncomment the following line:
  ##model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
  # Load pretrained model/tokenizer
  tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
  model = model_class.from_pretrained(pretrained_weights)
  with torch.cuda.device(cuda):
    for traincv, testcv in cv_data:
      torch.cuda.empty_cache()
      print("\n--------Fold {}--------\n".format(fold_count))
      # get the train and test from the dataset.
      X_train, X_test, y_train, y_test = X.iloc[traincv], X.iloc[testcv], y.iloc[traincv], y.iloc[testcv]
      train_essays = X_train['essay']
      #print("y_train",y_train)
      test_essays = X_test['essay']
      # model = model.cuda()
      #y_train = torch.tensor(y_train,dtype=torch.long)
      sentences = []
      tokenize_sentences = []
      train_bert_embeddings = []
      #bert_embedding = BertEmbedding()
      # for essay in train_essays:
      #   # get all the sentences from the essay
      #   sentences += essay_to_sentences(essay, remove_stopwords = True)
      # sentences = pd.Series(sentences)
      # print(train_essays)
      tokenized_train = train_essays.apply((lambda x: tokenizer.encode(x, add_special_tokens=True ,max_length=200)))
      tokenized_test = test_essays.apply((lambda x: tokenizer.encode(x, add_special_tokens=True ,max_length=200)))


      ## train
      max_len = 0
      for i in tokenized_train.values:
        if len(i) > max_len:
          max_len = len(i)
      padded_train = np.array([i + [0]*(max_len-len(i)) for i in tokenized_train.values])

      attention_mask_train = np.where(padded_train != 0, 1, 0)



      train_input_ids = torch.tensor(padded_train)
      train_attention_mask = torch.tensor(attention_mask_train)
      with torch.no_grad():
        last_hidden_states_train = model(train_input_ids, attention_mask=train_attention_mask)


      train_features = last_hidden_states_train[0][:,0,:].numpy()


      ## test
      max_len = 0
      for i in tokenized_test.values:
        if len(i) > max_len:
          max_len = len(i)
      padded_test = np.array([i + [0]*(max_len-len(i)) for i in tokenized_test.values])
      attention_mask_test = np.where(padded_test != 0, 1, 0)
      test_input_ids = torch.tensor(padded_test)
      test_attention_mask = torch.tensor(attention_mask_test)

      with torch.no_grad():
        last_hidden_states_test = model(test_input_ids, attention_mask=test_attention_mask)

      test_features = last_hidden_states_test[0][:,0,:].numpy()




      train_x,train_y = train_features.shape
      test_x,test_y = test_features.shape

      trainDataVectors = np.reshape(train_features,(train_x,1,train_y))
      testDataVectors = np.reshape(test_features,(test_x,1,test_y))

      lstm_model = get_model(bidirectional=False)
      lstm_model.fit(trainDataVectors, y_train, batch_size=128, epochs=70)
      y_pred = lstm_model.predict(testDataVectors)

      y_pred = np.around(y_pred)
      #y_pred.dropna()
      np.nan_to_num(y_pred)
      # evaluate the model
      result = cohen_kappa_score(y_test.values,y_pred,weights='quadratic')
      print("Kappa Score: {}".format(result))
      results.append(result)
      fold_count +=1
      import tensorflow as tf
      tf.keras.backend.clear_session()



  all_sets_score.append(results)
  print("Average kappa score value is : {}".format(np.mean(np.asarray(results))))
  set_count+=1
    # print(features.shape)

"""Below we will run for whole dataset using BERT but using CNN, which didnt performed so well."""

# For whole dataset
import time
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
cv = KFold(n_splits=5, shuffle=True)
# X = set8
# y = set8['domain1_score']
X= data
y = data['domain1_score']
cv_data = cv.split(X)
results = []
prediction_list = []
fold_count =1
# use_cuda = True
# if use_cuda and torch.cuda.is_available():
#   torch.cuda()
# Hyperpaprameters for LSTM
Hidden_dim1=300
Hidden_dim2=100
return_sequences = True
dropout=0.5
recurrent_dropout=0.4
input_size=768
activation='relu'
bidirectional = True
batch_size = 64
epoch = 100
#####
cuda = torch.device('cuda')
# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
## Want BERT instead of distilBERT? Uncomment the following line:
##model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
with torch.cuda.device(cuda):
  for traincv, testcv in cv_data:
    torch.cuda.empty_cache()
    print("\n--------Fold {}--------\n".format(fold_count))
    # get the train and test from the dataset.
    X_train, X_test, y_train, y_test = X.iloc[traincv], X.iloc[testcv], y.iloc[traincv], y.iloc[testcv]
    train_essays = X_train['essay']
    #print("y_train",y_train)
    test_essays = X_test['essay']
    # model = model.cuda()
    #y_train = torch.tensor(y_train,dtype=torch.long)
    sentences = []
    tokenize_sentences = []
    train_bert_embeddings = []
    #bert_embedding = BertEmbedding()
    # for essay in train_essays:
    #   # get all the sentences from the essay
    #   sentences += essay_to_sentences(essay, remove_stopwords = True)
    # sentences = pd.Series(sentences)
    # print(train_essays)
    tokenized_train = train_essays.apply((lambda x: tokenizer.encode(x, add_special_tokens=True ,max_length=55)))
    tokenized_test = test_essays.apply((lambda x: tokenizer.encode(x, add_special_tokens=True ,max_length=55)))


    ## train
    max_len = 0
    for i in tokenized_train.values:
      if len(i) > max_len:
        max_len = len(i)
    padded_train = np.array([i + [0]*(max_len-len(i)) for i in tokenized_train.values])

    attention_mask_train = np.where(padded_train != 0, 1, 0)



    train_input_ids = torch.tensor(padded_train)
    train_attention_mask = torch.tensor(attention_mask_train)
    with torch.no_grad():
      last_hidden_states_train = model(train_input_ids, attention_mask=train_attention_mask)


    train_features = last_hidden_states_train[0][:,0,:].numpy()


    ## test
    max_len = 0
    for i in tokenized_test.values:
      if len(i) > max_len:
        max_len = len(i)
    padded_test = np.array([i + [0]*(max_len-len(i)) for i in tokenized_test.values])
    attention_mask_test = np.where(padded_test != 0, 1, 0)
    test_input_ids = torch.tensor(padded_test)
    test_attention_mask = torch.tensor(attention_mask_test)

    with torch.no_grad():
      last_hidden_states_test = model(test_input_ids, attention_mask=test_attention_mask)

    test_features = last_hidden_states_test[0][:,0,:].numpy()
    train_x,train_y = train_features.shape
    test_x,test_y = test_features.shape
    trainDataVectors = np.reshape(train_features,(train_x,1,train_y))
    testDataVectors = np.reshape(test_features,(test_x,1,test_y))
    # print(trainDataVectors)
    # print(testDataVectors)
    # trainDataVectors = np.reshape(train_features,(train_x,train_y,1))
    # testDataVectors = np.reshape(test_features,(test_x,test_y,1))
    trainDataVectors = np.reshape(train_features,(train_x,1, train_y))
    testDataVectors = np.reshape(test_features,(test_x,1, test_y))
    # lstm_model = get_model_CNN(bidirectional=False, output_dims = 1)
    lstm_model = get_model(Hidden_dim1=Hidden_dim1, Hidden_dim2=Hidden_dim2, return_sequences=return_sequences,
                            dropout=dropout, recurrent_dropout=recurrent_dropout, input_size=input_size,
                            activation=activation, bidirectional=bidirectional)
    history = lstm_model.fit(trainDataVectors, y_train, batch_size=batch_size, epochs=epoch)
    plot_accuracy_curve(history)
    y_pred = lstm_model.predict(testDataVectors)
    y_pred = np.around(y_pred)
    #y_pred.dropna()
    np.nan_to_num(y_pred)
    # evaluate the model
    print(y_pred)
    result = cohen_kappa_score(y_test.values,y_pred,weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)
    fold_count +=1
    import tensorflow as tf
    tf.keras.backend.clear_session()
print("Average kappa score value is : {}".format(np.mean(np.asarray(results))))
  # print(features.shape)

import tensorflow as tf
tf.keras.backend.clear_session()

trainDataVectors = np.reshape(train_features,(train_x,train_y,1))
testDataVectors = np.reshape(test_features,(test_x,test_y,1))
print(y_train.shape)
print(trainDataVectors.shape)

# x = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
def plot_qwk_scores_all_sets():
  fig = plt.figure()
  ax = plt.subplot(111)
  x = [1,2,3,4,5]
  ax.plot(x, set1 , label='set1')
  ax.plot(x, set2, label='set2')
  ax.plot(x, set3, label='set3')
  ax.plot(x, set4, label='set4')
  ax.plot(x, set5, label='set5')
  ax.plot(x, set6, label='set6')
  ax.plot(x, set7, label='set7')
  ax.plot(x, set8, label='set8')
  plt.title('Set wise QWK using BERT for individual sets')
  ax.legend()
  plt.show()

"""Buildind word2vec model method to render the text and convert to word2vec feature vector."""

def build_word2vec(train_sentences, num_workers, num_features, min_word_count, context,
                     downsampling):
    model = Word2Vec(workers=num_workers, size=num_features, min_count=min_word_count, window=context,
                     sample=downsampling)
    # saving the word2vec model
    # model.wv.save_word2vec_format('word2vec_'+ str(fold_count) +'.bin', binary=True)
    cores = multiprocessing.cpu_count()
    print("\n {} cores using".format(cores))
    start_time = time.time()
    model.build_vocab(train_sentences, progress_per=10000)
    print('Time to build vocab using word2vec: {} sec'.format(time.time() - start_time))
    start_time = time.time()
    model.train(train_sentences, total_examples=model.corpus_count, epochs=epochs, report_delay=1)
    print('Time to train the word2vec model: {} mins'.format(time.time() - start_time))
    model.init_sims(replace=True)
    sorted_dic = sorted(top10.items(), key=lambda k: k[1], reverse=True)
    return model,sorted_dic

"""Below method will run on individual sets using word2vec"""

# Individual sets
import time
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
set_count = 1
all_sets_score = []
# Hyperparameters for word2vec
num_features = 400
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3
epochs = 30

# Hyperpaprameters for LSTM
Hidden_dim1=300
Hidden_dim2=100
return_sequences = True
dropout=0.5
recurrent_dropout=0.4
input_size=400
activation='relu'
bidirectional = True
batch_size = 64
epoch = 70
#####
####
import tensorflow as tf
tf.keras.backend.clear_session()

for s in sets:
  print("\n--------SET {}--------\n".format(set_count))
  set_count +=1
  X = s
  y = s['domain1_score']
  cv = KFold(n_splits=5, shuffle=True)
  #X, y = prepare_data(dataset_path=dataset_path)
  cv_data = cv.split(X)
  results = []
  prediction_list = []
  fold_count =1
  # hyperparameters for word2vec
  most_common_words= []
  print(X.shape)
  print(y.shape)
  for traincv, testcv in cv_data:
      print("\n--------Fold {}--------\n".format(fold_count))
      # get the train and test from the dataset.
      X_train, X_test, y_train, y_test = X.iloc[traincv], X.iloc[testcv], y.iloc[traincv], y.iloc[testcv]
      train_essays = X_train['essay']
      #print("y_train",y_train)
      test_essays = X_test['essay']
      #y_train = torch.tensor(y_train,dtype=torch.long)
      train_sentences = []
      # print("train_essay ",train_essays.shape)
      #print(X_train.shape,y_train.shape)
      for essay in train_essays:
          # get all the sentences from the essay
          train_sentences.append(essay_to_wordlist(essay, remove_stopwords = True))

      # word2vec embedding
      print("Converting sentences to word2vec model")
      model,_ = build_word2vec(train_sentences, num_workers, num_features, min_word_count, context,
                    downsampling)
      top10 = collections.defaultdict(int)

      # print("train_sentencesshap",len(train_sentences))
      trainDataVecs = np.array(getAvgFeatureVecs(train_sentences, model, num_features))
      test_sentences = []
      for essay_v in test_essays:
          test_sentences.append(essay_to_wordlist(essay_v, remove_stopwords=True))
      testDataVecs = np.array(getAvgFeatureVecs(test_sentences, model, num_features))
      trainDataVectors = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
      testDataVectors = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
      lstm_model = get_model_lstm(Hidden_dim1=Hidden_dim1, Hidden_dim2=Hidden_dim2, return_sequences=return_sequences,
                              dropout=dropout, recurrent_dropout=recurrent_dropout, input_size=input_size,
                              activation=activation, bidirectional=bidirectional)
      # print(trainDataVectors.shape)
      # print(y_train.shape)
      history = lstm_model.fit(trainDataVectors, y_train, batch_size=batch_size, epochs=epoch)
      plot_accuracy_curve(history)
      y_pred = lstm_model.predict(testDataVectors)
      y_pred = np.around(y_pred)
      np.nan_to_num(y_pred)
      result = cohen_kappa_score(y_test.values, y_pred, weights='quadratic')
      print("Kappa Score: {}".format(result))
      results.append(result)
      fold_count += 1

  print("Average kappa score value is : {}".format(np.mean(np.asarray(results))))
  all_sets_score.append(results)

"""Below method is for running the model on whole dataset using the word2vec model."""

# Whole Dataset Word2vec
X= data
y = data['domain1_score']
import time
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
set_count = 1
all_sets_score = []
# Hyperparameters for word2vec
num_features = 400
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3
epochs = 30

# Hyperpaprameters for LSTM
Hidden_dim1=300
Hidden_dim2=100
return_sequences = True
dropout=0.5
recurrent_dropout=0.4
input_size=400
activation='relu'
bidirectional = True
batch_size = 64
epoch = 70
#####
####
import tensorflow as tf
tf.keras.backend.clear_session()
cv = KFold(n_splits=5, shuffle=True)
#X, y = prepare_data(dataset_path=dataset_path)
cv_data = cv.split(X)
results = []
prediction_list = []
fold_count =1
# hyperparameters for word2vec
most_common_words= []
print(X.shape)
print(y.shape)
for traincv, testcv in cv_data:
    print("\n--------Fold {}--------\n".format(fold_count))
    # get the train and test from the dataset.
    X_train, X_test, y_train, y_test = X.iloc[traincv], X.iloc[testcv], y.iloc[traincv], y.iloc[testcv]
    train_essays = X_train['essay']
    #print("y_train",y_train)
    test_essays = X_test['essay']
    #y_train = torch.tensor(y_train,dtype=torch.long)
    train_sentences = []
    # print("train_essay ",train_essays.shape)
    #print(X_train.shape,y_train.shape)
    for essay in train_essays:
        # get all the sentences from the essay
        train_sentences.append(essay_to_wordlist(essay, remove_stopwords = True))

    # word2vec embedding
    print("Converting sentences to word2vec model")
    model,_ = build_word2vec(train_sentences, num_workers, num_features, min_word_count, context,
                  downsampling)
    top10 = collections.defaultdict(int)

    # print("train_sentencesshap",len(train_sentences))
    trainDataVecs = np.array(getAvgFeatureVecs(train_sentences, model, num_features))
    test_sentences = []
    for essay_v in test_essays:
        test_sentences.append(essay_to_wordlist(essay_v, remove_stopwords=True))
    testDataVecs = np.array(getAvgFeatureVecs(test_sentences, model, num_features))
    trainDataVectors = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
    testDataVectors = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
    lstm_model = get_model_lstm(Hidden_dim1=Hidden_dim1, Hidden_dim2=Hidden_dim2, return_sequences=return_sequences,
                            dropout=dropout, recurrent_dropout=recurrent_dropout, input_size=input_size,
                            activation=activation, bidirectional=bidirectional)
    # print(trainDataVectors.shape)
    # print(y_train.shape)
    history = lstm_model.fit(trainDataVectors, y_train, batch_size=batch_size, epochs=epoch)
    plot_accuracy_curve(history)
    y_pred = lstm_model.predict(testDataVectors)
    y_pred = np.around(y_pred)
    np.nan_to_num(y_pred)
    result = cohen_kappa_score(y_test.values, y_pred, weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)
    fold_count += 1

print("Average kappa score value is : {}".format(np.mean(np.asarray(results))))
