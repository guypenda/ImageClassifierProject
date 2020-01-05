# Imports here
import numpy as np
import pandas as pd

import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models

from collections import OrderedDict

from workspace_utils import active_session

import matplotlib.pyplot as plt

import seaborn as sb

import argparse
from time import time, sleep
from os import listdir

import os 
import sys

from PIL import Image

import json

from myfunctions import *
from myutils import *


def main():
    # Measures total program runtime by collecting start time
    #start_time = time()
    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    
    #Loading the model
    model, optimizer, epochs, train_categories = load_model(in_arg.checkpoint)
    
    #Retrieve categories
    with open(in_arg.category_names, 'r') as f:
        categories = json.load(f)    
    
    #Run the prediction
    probs, classes = predict(in_arg.img_path, model, True, in_arg.top_k)
    
    # get classes names
    class_names = []
    for i in range(len(classes)):
        class_names.append(categories.get(str(classes[i])))
    
    print('Flower real name: {}'.format(get_name(in_arg.img_path, categories)))
    print('-------------------------')
    print('Predicted names:')
    print(*class_names, sep=', ')
    print('-------------------------')
    print('Predicted Probabilities:')
    
    print(*probs, sep=', ')
    print('-------------------------')

# Functions defined below
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. 
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()

    # Creates command line arguments
    parser.add_argument('img_path', type=str, default='', 
                        help='path to the image to classify')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth', 
                        help='path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=3, 
                        help='number of top percentages to consider')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='file name of the mapping between category code and names')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='Specify if the GPU shall be use or not. Default is yes')
    # returns parsed argument collection
    return parser.parse_args()

# Call to main function to run the program
if __name__ == "__main__":
    main()