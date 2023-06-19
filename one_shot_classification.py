from builtins import print
from os import makedirs
from xml.dom.minidom import Identified
import tensorflow as tf

import sys
import os
import argparse

from matplotlib import pyplot as plt
import cv2
import numpy as np

import pickle
import time

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--input_image_path", type=str, default='',
                        help="input image path")
    parser.add_argument("--saved_model_dir", type=str, default='',
                        help="Where restore model parameters from.")
    parser.add_argument("--query_image_path", type=str, default='',
                        help="query image path")
    parser.add_argument('--unk_threshold', type=float, default=0.016, 
                        help='minimum value to be classified as unknown.')
    

    return parser.parse_args()

def preprocess_input(input_image, shape=(224, 224), dtype=np.float32, gs=False):
    '''
    Reads and preprocesses the input image.
    Arguments ::
        input_image -- ndarray or str | image to be normalized or path to the image
        shape -- tupple | shape to resize the input image in the form (w, h) | default None
                    | if None, does not resize the input image
        dtype -- datatype of the input to the model | default np.float32
        gs -- boolean | default False | when True converts image to gs
    Returns ::
        input_image -- ndarray | preprocessed input image
    '''
    if isinstance(input_image, str):
        ## Read the image
        # print(input_image)
        input_image = cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB)
    
    if gs:
        input_image = np.ones_like(input_image) * cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]

    if dtype == np.int8:
        return input_image[np.newaxis, :, :, :].astype(dtype)

    if np.max(input_image) > 1:
        input_image = input_image / 255.

    if shape != None:
        if (input_image.shape[0] != shape[1]) and (input_image.shape[1] != shape[0]):
            input_image = cv2.resize(input_image, shape)

    input_image = input_image[np.newaxis, :, :, :].astype(np.float32)

    return input_image

@tf.function
def test_step(X, model):
    '''
    Train one minibatch
    '''
    embeddings = model.call(X)

    return embeddings


def get_embeddings(input_image, model):
    '''
    This function infers one image and saves the output.
    Arguments --
        input_image_path -- str | input image path
        model - tf.keras.lauers.Model | pretrained model
    '''
    ## Pre-process image
    # input_image = preprocess_input(input_image_path, shape=(224, 224), gs=True)
    input_image = preprocess_input(input_image, shape=(224, 224))

    ## Infer
    embeddings = test_step(input_image, model).numpy()

    return embeddings

def one_shot_match(input_image, query_image, model, from_query_embed = False):    
    input_embed = get_embeddings(input_image, model)
    if not from_query_embed:
        query_embed = get_embeddings(query_image, model)
    else:
        query_embed = query_image

    dist = np.sum(np.sqrt((query_embed - input_embed) ** 2))

    print(dist)
    return dist < 3.2

if __name__ == '__main__':
    ## get all the arguments
    args = get_arguments()
    input_image_path = args.input_image_path
    query_image_path = args.query_image_path
    saved_model_dir = args.saved_model_dir
    unk_threshold = args.unk_threshold

    ## reset tf graph
    tf.keras.backend.clear_session()    

    ## allow gpu growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    gpus = ['gpu:'+gpu.name[-1] for gpu in gpus]
    print(f'GPUs : {gpus}')

    ## instantiate the model
    print('Loading Model...', end='\r')
    model = tf.saved_model.load(saved_model_dir)

    one_shot_match(input_image_path, query_image_path, model)
        



