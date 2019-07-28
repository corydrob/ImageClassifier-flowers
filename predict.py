#load a network
#predict classification of an image based upon that loaded trained network
import numpy as np
import helper
import torch
import argparse
import json
from PIL import Image

parser = argparse.ArgumentParser(description='Predict a flower species using a neural net.')

parser.add_argument('image_path', type=str,
                    help='path to image to predict classification of.')
parser.add_argument('checkpoint', type=str,
                    help='path to checkpoint file containing information on a trained image classifier')
parser.add_argument('--topk', type=int, default=5,
                    help='number of predictions to list, starting with highest-odds prediction and going in order.')
parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                    help='mapping of image labels to human-readable names.')
parser.add_argument('--gpu', nargs='?', const=1,
                    help='train on gpu.')

args = parser.parse_args()

def predict(image_path, checkpoint, device, categories_to_names, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(categories_to_names, 'r') as f:
        cat_to_name = json.load(f)
    # TODO: Implement the code to predict the class from an image file
    img = helper.process_image(image_path)
    model, class_to_idx = helper.load_checkpoint(checkpoint)#,optimizer=load_checkpoint(model)
    
    img = torch.Tensor(img)
    
    #unsqueezing recommended on torch discussion forums for error i was having at:
    #https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list-of-1-values-to-match-the-convolution-dimensions-but-got-stride-1-1/17140
    img=img.unsqueeze(0)
    
    img = img.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        log_prob=model(img)
        probabilities = torch.exp(log_prob)
        probs, classes=probabilities.topk(topk,dim=1)
    classes = classes.cpu()
    classes = classes.numpy()
    classification = {number: string for string, number in class_to_idx.items()}
    class_names = [cat_to_name[classification[item]] for item in classes[0]]
    probs = probs.cpu()
    probs = probs.numpy()
    return probs[0],class_names


if args.gpu:
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

probs, class_names = predict(args.image_path, args.checkpoint,device,args.category_names,args.topk)

results=''

for i in range(len(probs)):
    row = class_names[i]+': '+ str(probs[i])+'\n'
    results+=row
print(results)