#defining model for use in train.py
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from collections import OrderedDict
import numpy as np
from PIL import Image

def create_model(hidden_units=2048,arch='vgg16',learnrate=0.001):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_units=25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_units=9216
    else:
        print('Defaulting to vgg16.')
        model = models.vgg16(pretrained=True)
        input_units=25088

    for param in model.parameters():
        param.requires_grad=False

    classifier = nn.Sequential(OrderedDict([
                              ('hidden_layer_1', nn.Linear(input_units, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dropout1',nn.Dropout(0.2)),
                              ('hidden_layer_2', nn.Linear(hidden_units, int(hidden_units/4))),
                              ('relu2', nn.ReLU()),
                              ('output', nn.Linear(int(hidden_units/4),102)),
                              ('log_softmax', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier=classifier
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(),lr=learnrate)
    return model, criterion, optimizer
