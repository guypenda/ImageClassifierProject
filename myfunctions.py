# Imports here
import numpy as np
import pandas as pd

import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models

from collections import OrderedDict

from myutils import *


def create_model(n_outputs, model_architecture='vgg16', hidden_units=1000):

    # Load a pretrained neural network
    if(model_architecture=='vgg16'):
        model = models.vgg16(pretrained = True)
    elif model_architecture=='vgg13':
        model = models.vgg13(pretrained = True)
    elif model_architecture=='vgg19':
        model = models.vgg19(pretrained = True)
    elif model_architecture.find('resnet') >= 0:
        model = models.resnet50(pretrained = True)
    else:
        print('No model specified or the specified model is not taken into account. The vgg16 will be used instead')
        model = models.vgg16(pretrained = True)

    # Disable the backpropagation for all the model's layers
    for param in model.parameters():
        param.requires_grad = False

    # create my own classifier. 
    # get the inputs from the former classifier
    if model_architecture.find('resnet') >= 0:
        n_inputs = model.fc.in_features
    else:
        n_inputs = model.classifier[0].in_features
    # and we have 102 flowers category to classify
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(n_inputs, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('do2', nn.Dropout(0.5)),
                              ('fc3', nn.Linear(hidden_units, n_outputs)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    if model_architecture.find('resnet') >= 0:
        model.fc = classifier
    else:
        model.classifier = classifier
    
    return model, classifier


# Define the validation function 
def validation(model, validloader, criterion, device):
    validation_loss = 0
    accuracy = 0
    for images, labels in validloader:
        
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        validation_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validation_loss, accuracy


# Defining the training function
def train_model(model, trainloader, validloader, device, criterion, optimizer, epochs=20, print_every=10):
    steps = 0
    running_loss = 0
    model.to(device)
    for e in range(epochs):
        model.train() # Put the model in training mode
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)                
        
            optimizer.zero_grad()
        
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval() # This is to improve the performances and use less machine resources
            
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                     test_loss, accuracy = validation(model, validloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
            
            # Make sure training is back on for the next steps and epochs
            model.train()
    return model, optimizer, epochs


def test_model(model, testloader, device='cuda'):

    # Load the model to CUDA
    model.to(device)
    # Put the model to evaluation mode
    model.eval()
    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device) 

    # Initialize the model accuracy to 0
    accuracy = 0
    
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logits = model.forward(images)
        
    ps = torch.exp(logits)
    equality = (labels.data == ps.max(dim=1)[1])
    accuracy += equality.type(torch.FloatTensor).mean()
    print("Model accuracy on test dataset : {}% ".format(accuracy * 100))

    return model, accuracy


def save_model(model, model_name, epochs, optimizer, path='checkpoint.pth', train_categories=None):
    # bring the model back to the cpu in case it's on cuda
    model.to('cpu')
    
    checkpoint = {'epochs': epochs,
                  'model_arch': model_name,
              'model_state_dict': model.state_dict(),
              'model_classifier': model.fc if model_name.find('resnet') >= 0 else model.classifier,
              'classifier_state_dict': model.fc.state_dict if model_name.find('resnet') >= 0 else model.classifier.state_dict,
              'optimizer': optimizer,
              'optimizer_state_dict': optimizer.state_dict(),              
              'categories': train_categories,
             }
    torch.save(checkpoint, path)
    

def load_model(checkpoint_path):
    model = None
    optimizer = None
    class_to_idx = None
    epochs = None
    
    # Load the check point
    checkpoint = torch.load(checkpoint_path)
    model_architecture = checkpoint.get('model_arch')
    
    # Load a pretrained neural network
    if(model_architecture=='vgg16'):
        model = models.vgg16(pretrained = True)
    elif model_architecture=='vgg13':
        model = models.vgg13(pretrained = True)
    elif model_architecture=='vgg19':
        model = models.vgg19(pretrained = True)
    elif model_architecture.find('resnet') >= 0:
        model = models.resnet50(pretrained = True)
    else: 
        model = models.vgg16(pretrained = True)
        
    # Disable the backpropagation for all the layers
    for param in model.parameters():
        param.requires_grad = False
    
    model.state_dict = checkpoint.get('model_state_dict')
    
    if model_architecture.find('resnet') >= 0:
        model.fc = checkpoint.get('model_classifier')
    else:
        model.classifier = checkpoint.get('model_classifier')
      
    
    if model_architecture.find('resnet') >= 0:
        model.fc.state_dict = checkpoint.get('classifier_state_dict')
    else:
        model.classifier.state_dict = checkpoint.get('classifier_state_dict')
    
    
    # Create a new Adam optimizer 
    optimizer = checkpoint.get('optimizer')
    optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))
    
    # Retrieve the flowers categories in case it's saved
    train_categories = checkpoint.get('categories')
    
    # Retrieve the number of epochs
    epochs = checkpoint.get('epochs')
    
    return model, optimizer, epochs, train_categories


def predict(image_path, model, gpu=True, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''        
    image = process_image(image_path)
    
    # Choose the correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)     
    image = image.to(device)
    
    # Turn off gradients to speed up this part
    with torch.no_grad():
        model.eval()
        logits = model.forward(image.unsqueeze_(0))
        ps = torch.exp(logits)
        
    prob, classes = ps.topk(topk, dim=1)
    
    if gpu:
        prob = prob.to('cpu')
        classes = classes.to('cpu')
        
    return prob.reshape((topk)).numpy(), classes.reshape((topk)).numpy()
