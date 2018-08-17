import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import config
import functions
import PIL.ImageOps

from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.optim import lr_scheduler
from skimage import io, transform
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix

def build_model(model_name=None, model_path=None, fc_flag=1, num_of_feature=5, optimizer=None):
    if model_name:
        model_ft = model_name
    else:
        model_ft = torch.load(model_path)
        return model_ft
    if fc_flag == 1:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_of_feature)
        model_ft.fc = model_ft.fc.to(config.device)
    else:
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_of_feature)
        model_ft.classifier = model_ft.classifier.to(config.device)
    model_ft = model_ft.to(config.device)
#    criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([0.01, 0.03, 0.2, 0.35, 0.4])).to(config.device)
    criterion = nn.CrossEntropyLoss().to(config.device)
    optimizer = optim.Adam(model_ft.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return model_ft, criterion, optimizer, exp_lr_scheduler

def train_model(model, criterion, optimizer, scheduler, num_epochs=20, dataloader=None, datasizes=None, load_dir=None, save_dir="outputs/undefined.pt", labels=None, rotate_flag=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    if load_dir:
        model = torch.load(load_dir)
        
    best_acc = 0.0

    for epoch in range(num_epochs):
        if config.checkpoint_flag == 1:
            if config.start_epoch:
                epoch = config.start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            counter = 0
            # Iterate over data.
            for batch in dataloader[phase]:
                inputs = batch['image'].float().to(config.device)
                labels = batch['etiketler'].to(config.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                counter += 1

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    #loss = criterion(outputs, labels)
                    
                    soft_outputs = torch.nn.Softmax()(outputs)
                    value_preds, preds = torch.max(soft_outputs, 1)
                    if rotate_flag:
                        for i in [90, 180, 270]:
                            new_inputs = batch["image"].float().to("cpu")
                            for image in range(len(new_inputs)):
                                a = transforms.ToPILImage()(new_inputs[image])
                                a = PIL.ImageOps.invert(a)
                                a = a.rotate(i)
                                a = transforms.functional.to_grayscale(a, 3)
                                a = transforms.ToTensor()(a)
                                new_inputs[image] = a
                        new_outputs = model(new_inputs.to(config.device))
                        soft_new_outputs = torch.nn.Softmax()(new_outputs)
                        value_new_preds, new_preds = torch.max(soft_new_outputs, 1)

                        for l in range(len(value_new_preds)):
                            if value_new_preds[l] > value_preds[l]:
                                preds[l] = new_preds[l]
                                value_preds[l] = value_new_preds[l]
                                outputs[l] = new_outputs[l]
                    loss = criterion(outputs, labels)    
                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        print("{} / {} batch okunuyor".format(counter, len(dataloader[phase])))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
#            print(confusion_matrix(labels.data, preds, labels))

            epoch_loss = running_loss / datasizes[phase]
            epoch_acc = running_corrects.double() / datasizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print("Running_corrects: {}".format(running_corrects))
            with open("logs/log" + (str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')), "a") as dosya:
                dosya.write("{} Epoch: {}, Acc: {}, Loss: {}\n".format(phase, epoch, epoch_acc, epoch_loss))
                dosya.flush()

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, save_dir)
            
				
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, criterion, dataloader=None, datasizes=None, labels_for_confusion=None, rotate_flag=False ):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for batch in dataloader["test"]:
        torch.no_grad()
        inputs = batch["image"].float().to(config.device)
        labels = batch["etiketler"].to(config.device)
          
        #print(type(inputs), inputs.shape, inputs)

        outputs = model(inputs)
        soft_outputs = torch.nn.Softmax()(outputs)
        value_preds, preds = torch.max(soft_outputs, 1)

        if rotate_flag:
            for i in [90,180,270]:
                print("{} derece donduruluyor".format(i))
                new_inputs = batch["image"].float().to("cpu")
                # rotate input batch
                for image in range(len(new_inputs)):   
                    a = transforms.ToPILImage()(new_inputs[image]) 
                    a = PIL.ImageOps.invert(a)
                    a = a.rotate(i)
                    a = transforms.functional.to_grayscale(a, 3)
                    a = transforms.ToTensor()(a)
                    new_inputs[image] = a
            new_outputs = model(new_inputs.to(config.device))
            soft_new_outputs = torch.nn.Softmax()(new_outputs)
            value_new_preds, new_preds = torch.max(soft_new_outputs, 1)
            
            for l in range(len(value_new_preds)):
                if value_new_preds[l] > value_preds[l]:
                    preds[l] = new_preds[l]
                    value_preds[l] = value_new_preds[l]
                    outputs[l] = new_outputs[l]
            

        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    #Print confusion matrix
    #confusion_matrix(labels.data, preds, labels_for_confusion)

    epoch_loss = running_loss / datasizes["test"]
    epoch_acc = running_corrects.double() / datasizes["test"]
    print("{} Loss: {:.4f} Acc: {:.4f}".format("test", epoch_loss, epoch_acc))
