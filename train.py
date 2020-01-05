

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
    start_time = time()
    
    # Creates & retrieves Command Line Arugments
    in_arg = get_input_args()
    
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Load the data
    print('Loading data....')
    data_transforms, image_datasets, dataloaders = load_data(train_dir, valid_dir, test_dir, batch_size=168)
   
    
    #Map the categories
    n_outputs, cat_to_name, train_images, valid_images, test_images = label_map(train_dir, valid_dir, test_dir, categories_file='cat_to_name.json')
    print('Data Loaded!')
    print('--------------------------------\n\n')
    #print(data_transforms)
    
    #Build the model
    print('Initializing a {} model...'.format(in_arg.arch))
    model, claissifier = create_model(n_outputs, in_arg.arch, in_arg.hidden_units)
    
    #Fetching the cuda server
    device = torch.device("cuda:0" if torch.cuda.is_available() and in_arg.gpu else "cpu")
    
    # Setting the criterion, the learn rate and traning only the classifier
    learnrate = in_arg.learning_rate
    criterion = nn.NLLLoss()
    if(in_arg.arch.find('vgg')>=0):
        optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    else:
       optimizer = optim.Adam(model.fc.parameters(), lr=learnrate)
    print('Model initialized!')
    print('--------------------------------\n\n')
    
    print('Starting model training...')
    #Train the model
    with active_session():
        model, optimizer, epochs = train_model(model, dataloaders['train'], dataloaders['valid'], device, criterion, optimizer, epochs=in_arg.epochs, print_every=20)
    print('Model Trained after {} epochs!'.format(epochs))
    print('--------------------------------\n\n')
    
    print('Validating the model on test data set...')
    # Evaluate the model accuracy on test dataset
    model, accuracy = test_model(model, dataloaders['test'], device=device)
    print('--------------------------------\n\n')
    
    print('Saving the model...')
    # Save the model    
    save_model(model, in_arg.arch, epochs, optimizer, in_arg.arch + '_model_gpe.pth', cat_to_name)
    print('Model saved at {}'.format(in_arg.arch + '_model_gpe.pth'))
    print('--------------------------------\n\n')
    
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
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
    parser.add_argument('data_dir', type=str, default='flowers', 
                        help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='.', 
                        help='path to folder to save the model')
    parser.add_argument('--arch', type=str, default='vgg16', 
                        help='chosen model')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate of the model')
    parser.add_argument('--hidden_units', type=int, default=1000,
                        help='Number of hidden units of the classifier')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train the model')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='Specify if the GPU shall be use or not. Default is yes')
    # returns parsed argument collection
    return parser.parse_args()


# Call to main function to run the program
if __name__ == "__main__":
    main()