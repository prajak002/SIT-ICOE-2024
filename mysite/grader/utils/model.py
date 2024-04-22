from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.backend as K



def get_model():
    """Define the model."""
    model = Sequential()
    model.add(LSTM(200, dropout=0.4, recurrent_dropout=0.4,
                   input_shape=[1, 200], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop', metrics=['mae'])
    model.summary()

    return model
