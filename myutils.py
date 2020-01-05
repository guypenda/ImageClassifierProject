# Imports here
import numpy as np
import pandas as pd

import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models

from collections import OrderedDict

import os 

from PIL import Image

import json

def load_data(traindir, validdir, testdir, batch_size=100):
    """
    Load project data/images from a directory.
    Parameters:
     traindir - The directory containing the training data
     validdir - The directory containing the validation data
     testdir - The directory containing the testing data
     batch_size - The size of every batch of data in the datasets
    Returns:
     data_transforms -directory of transformations applied to the data  
    image_datasets -directory of image datasets with applied transformations
     dataloaders -directory of dataloaders containing the batches of train, validation and test data
    """
    # Keep the transforms in a dictionary
    data_transforms = {
        # Training dataset transforms
        'train' : transforms.Compose([transforms.RandomRotation(30),
                                      transforms.Resize(256),
                                      transforms.ColorJitter(),
                                      transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                     ]),
        # Validation and testing transforms
        'test' : transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])
                                    ]) 
    }

    # DONE: Load the datasets with ImageFolder
    image_datasets = {
        'train' : datasets.ImageFolder(traindir, transform=data_transforms.get('train')),
        'valid' : datasets.ImageFolder(validdir, transform=data_transforms.get('test')),
        'test'  : datasets.ImageFolder(testdir, transform=data_transforms.get('test'))
    }

    # DONE: Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets.get('train'), batch_size=batch_size, shuffle=True),
        'valid' : torch.utils.data.DataLoader(image_datasets.get('valid'), batch_size=batch_size, shuffle=True),
        'test' : torch.utils.data.DataLoader(image_datasets.get('test'), batch_size=batch_size, shuffle=True)
    }

    return data_transforms, image_datasets, dataloaders


def label_map(traindir, validdir, testdir, categories_file='cat_to_name.json'):
    """
    Load the real images categories with the associated category codes
    Parameters:
     traindir - name of the directory containing the training data
     validdir - name of the directory containing the validation data
     testdir - name of the directory containing the testing data
     categories_file - json file name. File containing the mapping between every category code and the corresponding category name
    Returns:
     n_outputs - the number of available categories 
     cat_to_name -the dictionary containing the mapping between every category code and the corresponding category name  
     train_images -directory of train image with their category code
     valid_images -directory of validation image with their category code
     test_images -directory of test image with their category code
    """
    with open(categories_file, 'r') as f:
        cat_to_name = json.load(f)

    # The number of different category in the 'cat_to_name' file should be the number of output of our n       
    n_outputs = len(cat_to_name.keys())

    # Saves images with categories
    train_images = dict()   
    valid_images = dict()
    test_images = dict()

    # Iterate through each category
    for cat in os.listdir(traindir):    
    
        #Save training images with their categories
        train_imgs = os.listdir(traindir + '/' + cat)
        train_images[cat] = train_imgs 

        #Save validation images with their categories
        valid_imgs = os.listdir(validdir + '/' + cat)
        valid_images[cat] = valid_imgs 

        #Save testing images with their categories
        test_imgs = os.listdir(testdir + '/' + cat)
        test_images[cat] = test_imgs 

    return n_outputs, cat_to_name, train_images, valid_images, test_images


def get_name(image_path, cat_catalogue):
    ''' From an image path and the dictionary containing the corresponding image class name and class code, return the image class name.
    ''' 
    img_class = image_path.split('/')[-2]
    return cat_catalogue[img_class]

def get_random_img(adir='flowers/test'):
    ''' Pick a random image from a directory'''
    sep = os.path.sep
    class_id = np.random.choice(os.listdir(adir))
    img_name = np.random.choice(os.listdir(adir + sep + class_id))
    return adir + sep + class_id + sep + img_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Open the image
    img = Image.open(image)
    
    # Retrive the original size
    width, height = img.size
    
    # Compute the new size considering the shortest side to be 256 pixels and keeping the aspect ratio
    if(width > height):
        new_width = int(256*width/height)
        new_height = 256        
    else:
        new_width = int(256*width/height)
        new_height = 256
    
    # Resize the image
    img = img.resize((new_width, new_height))
    
    # Center crop the image
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    # We divide by 256 as 0-255 is 256 values
    np_img = np.array(img) / 256

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    # Normalizing the image
    np_img -= means

    np_img /= stds
    
    return torch.Tensor(np_img.transpose())


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax