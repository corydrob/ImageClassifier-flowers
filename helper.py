#save and load functions
#image processing functions
import torch
from torchvision import models
import numpy as np
from PIL import Image

def save_model(model, train_set):
    checkpoint = {'training_indices': train_set.class_to_idx,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')
    
def load_checkpoint(file, arch='VGG'):
    checkpoint=torch.load(file)
    if arch=='VGG':
        model = models.vgg16(pretrained=True)
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        return 'what'
    
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['training_indices']
    
    for param in model.parameters():
        param.requires_grad=False 
    #TODO: debug optimizer in save file    
    #optimizer = torch.optim.Adam(model.classifier.parameters(),lr=0.001)
    #optimizer.load_state_dict(checkpoint['opt_state_dict'])
    #TODO: figure out why class_to_idx was not working properly.
    return model, model.class_to_idx#, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    color_means=[0.485, 0.456, 0.406]
    color_stdevs = [0.229, 0.224, 0.225]
    cropped_size = 224
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    #recording original image size
    width0, height0 = pil_image.size
    #print('original dimensions: {}, {}'.format(width0,height0))
    
    #my code failed the sanity check. Experimenting with changes here.
    #dumb things I tried preserved as comments for my reference.
    #looks like resize breaks aspect ratios if you're not careful, so that's 1st mistake
    
    #pil_image = pil_image.resize((255,255))
    #pil_image = pil_image.crop((16,16,239,239))
    #didn't work to just call thumbnail. smallest dimension was < 256
    #thumbnail documentation says "no larger than the given size"->set one dim to 
    #pil_image.thumbnail((256,256))
    
    if width0>height0:
        pil_image.thumbnail((width0*10,256))
    else:
        pil_image.thumbnail((256,height0*10))
    #print(pil_image.size)
    #Computing 4 coords required for PIL.Image.crop
    #need to crop based on resized dimensions of image
    width, height = pil_image.size
    left = (width-cropped_size)/2
    upper = (height-cropped_size)/2
    right = left+cropped_size
    lower = upper+cropped_size
    
    pil_image = pil_image.crop((left,upper,right,lower))
    
    #convert to np.array for normalization.
    np_image = np.array(pil_image)
    np_image = np_image/255
    np_image = np_image-color_means
    np_image = np_image/color_stdevs
    
    return np_image.transpose(2,0,1)