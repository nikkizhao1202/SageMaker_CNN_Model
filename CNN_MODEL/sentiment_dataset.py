"""
Load Data Set 

@Author: Nikk Zhao
"""
import os
import json
import math
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import boto3

# from tf.data import Dataset



def train_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "train")

def validation_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "validation")

def eval_input_fn(training_dir, config):
    return _input_fn(training_dir, config, "eval")

def serving_input_fn(_, config):
    # Here it concerns the inference case where we just need a placeholder to store
    # the incoming images ...
    tensor = tf.placeholder(dtype=tf.float32, shape=[1, config["embeddings_vector_size"]])
    inputs = {config["input_tensor_name"]: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def _load_json_file(json_path, config):

    features = []
    labels = []
    
    s3 = boto3.resource('s3') 
    obj = s3.Object('twitter-text', json_path)
    file = obj.get()['Body'].read().decode()
    
    for line in file.split('\n'):

        entry = json.loads(line)

        if len(entry["features"]) != config["padding_size"]:
            raise ValueError(
                "The size of the features of the entry with twitterid {} was not expected".format(
                    entry["twitterid"]))

        labels.append(float(entry["sentiment"]) / 4)
        features.append(entry["features"])

    return features, labels

def _input_fn(directory, config, mode):

    print("Fetching {} data...".format(mode))
    
    client = boto3.client('s3')

    all_files = client.list_objects_v2(Bucket='twitter-text', Prefix= directory)['Contents']
    
    all_features = []
    all_labels = []
    
    for i in range(1,len(all_files)):
        features, labels = _load_json_file(all_files[i]['Key'], config)
        all_features += features
        all_labels += labels
        # print(len(all_features))

    num_data_points = len(all_features)
    num_batches = math.ceil(len(all_features) / config["batch_size"])

    dataset = tf.data.Dataset.from_tensor_slices((all_features, all_labels))

    if mode == "train":

        dataset = tf.data.Dataset.from_tensor_slices((all_features, all_labels))
        dataset = dataset.batch(config["batch_size"]).shuffle(10000, seed=12345).repeat(
            config["num_epoch"])
        num_batches = math.ceil(len(all_features) / config["batch_size"])

    if mode in ("validation", "eval"):

        dataset = dataset.batch(config["batch_size"]).repeat(config["num_epoch"])
        num_batches = math.ceil(len(all_features) / config["batch_size"])

    iterator = dataset.make_one_shot_iterator()
    dataset_features, dataset_labels = iterator.get_next()

    return [{config["input_tensor_name"]: dataset_features}, dataset_labels,
            {"num_data_point": num_data_points, "num_batches": num_batches}]



