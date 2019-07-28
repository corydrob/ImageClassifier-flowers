#train a network
#save its checkpoint for later use
from model_creation import create_model
import helper
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import argparse
from workspace_utils import active_session


parser = argparse.ArgumentParser(description='Train a neural net on flower photos')

parser.add_argument('data_dir', type=str,
                    help='main directory for image data.')
parser.add_argument('--arch', type=str, default = 'vgg16',
                    help='Which pretrained classifier to use')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help='learning rate to train classifier with')
parser.add_argument('--hidden_units', type=int, default = 2048,
                    help='units for hidden layer in classifier.')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train for')
parser.add_argument('--gpu', nargs='?', const=1,
                    help='train on gpu.')

args = parser.parse_args()
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
model,criterion,optimizer = create_model(args.hidden_units, args.arch,args.learning_rate)

if args.gpu:
    device=torch.device('cuda')
else:#args.gpu==0:
    device=torch.device('cpu')
'''
else:
    print('gpu preference not specified, using GPU if available.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
print('Using device: ',device)
model.to(device)
#prepare data for training
#Data augmentation performed on training set
training_transform = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# Load the datasets with ImageFolder
train_set = datasets.ImageFolder(train_dir,transform=training_transform)
valid_set = datasets.ImageFolder(valid_dir, transform=valid_transform)

# DataLoaders using each set/ transform.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True)

epochs=args.epochs
steps = 0
train_losses, valid_losses = [],[]
validation_accuracy = 0 
validity_threshold = 0.8
validate_every = 10

#with active_session():
for e in range(epochs):
    running_loss=0

    #go through training set
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        steps += 1
        #reset gradient to zero
        optimizer.zero_grad()

        #run model, compute loss
        log_probabilities = model(images)
        loss = criterion(log_probabilities,labels)
        #backprop and step
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % validate_every == 0:
            valid_loss = 0
            validation_accuracy = 0

            #gradient goes off & model goes into eval mode b/c we don't want to train when validating. 
            model.eval()
            with torch.no_grad():
                for images,labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    #forward pass and loss calculation for validation set
                    log_probabilities = model(images)
                    valid_loss+=criterion(log_probabilities,labels)

                    #compute accuracy
                    probabilities = torch.exp(log_probabilities)
                    top_probs, top_class = probabilities.topk(1,dim=1)
                    equality = top_class==labels.view(*top_class.shape)
                    validation_accuracy += torch.mean(equality.type(torch.FloatTensor))
            model.train()

            #save information about losses over time in case it winds up useful for debugging
            train_losses.append(running_loss/len(train_loader))
            valid_losses.append(valid_loss/len(valid_loader))

            #print training stats every time we validate
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                  "Validation Accuracy: {:.3f}".format(validation_accuracy/len(valid_loader)))

            #If this validation pass confirmed our accuracy is high, break once to escape the inner for-loop
            if (validation_accuracy/len(valid_loader))>=validity_threshold:
                break
    #....then break again to escape the next for-loop
    if (validation_accuracy/len(valid_loader))>=validity_threshold:
            break

helper.save_model(model,train_set)