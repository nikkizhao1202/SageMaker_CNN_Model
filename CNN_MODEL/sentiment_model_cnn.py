"""
Model definition for CNN sentiment training
@Author: Nikk Zhao
"""

import os
import tensorflow as tf
import numpy as np
import boto3
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam




def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """

    '''
    layer1: embedding layer
    '''
    input_length = config["padding_size"]
    input_dim = config["embeddings_dictionary_size"]
    output_dim = config["embeddings_vector_size"]
       
    client = boto3.client('s3')
    s3 = boto3.resource('s3') 
    obj = client.get_object(Bucket='twitter-text', Key='training/data/glove.twitter.27B.25d.txt')
    contents = obj['Body'].read().decode()
    glove = contents.split('\n')

    embedding_matrix = np.zeros((input_dim, output_dim))

    for i in range(input_dim):
        if len(glove[i].split()[1:]) != output_dim :
            continue
        embedding_matrix[i] = np.asarray(glove[i].split()[1:], dtype='float32')


    embedding_layer = Embedding(input_dim, output_dim, weights=[embedding_matrix],input_length=input_length, trainable=True)

    cnn_model = Sequential() 
    cnn_model.add(embedding_layer)


    """
    Layer2: Convolution1D layer
    """
    cnn_model.add(Convolution1D(
        filters=100,
        kernel_size=2,
        input_shape=(input_length, input_dim),
        strides=1,
        padding='valid',
        activation='relu'
    ))

    """
        Layer3: GlobalMaxPool1D layer
        """
    cnn_model.add(GlobalMaxPooling1D())
    """
    Layer4: Dense layer
    """
    cnn_model.add(Dense(units=100, activation='relu'))
    """
    Layer5: Dense layer
    """
    cnn_model.add(Dense(units=1, activation='sigmoid'))
    opt = Adam()
    cnn_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving
    """

    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))

