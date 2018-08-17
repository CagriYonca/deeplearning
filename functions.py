from __future__ import print_function, division
import os
import torch
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
import config

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils
from PIL import Image

# Veriyi goster
def show_data(n):
    img = mpimg.imread("images/{}".format(img_name[n])) # images klasorundeki resimleri tek tek aliyor
    imgplot = plt.imshow(img)                           # bu resmi cizdiriyor
    plt.show()                                          # bunu ekranda gosteriyor
    print(img_label[n])                                 # resmin etiketini alta yazdiri  

def show_labeled_batch(sample_batched):                                                         # etiketli veriyi goster 
    images_batch, documents_batch = sample_batched["image"], sample_batched["etiketler"]        # resim batchini ve etiket batchini al
    batch_size = len(images_batch)                                                              # batch size resim batchinin uzunlugundan al
    im_size = images_batch.size(2)                                                              # resim size' batch'in size'n 2.parametresinden
                                                                                                # al
    grid = utils.make_grid(images_batch)                                                        # image batch'lerini kafese al
    plt.imshow(grid.numpy().transpose((1,2,0)))                                                 # kafesi cizdir

    for i in range(batch_size):
        plt.scatter(documents_batch[i, :, 0].numpy() + i * im_size,
                documents_batch[i, :, 1].numpy(),
                s=10, marker=".", c="r")
        plt.title("Batch from dataloader")

def save_checkpoint(state, is_best, filename="outputs/checkpoint.pt"):
    if is_best:
        print("Saving a new best")
        torch.save(state, filename)
    else:
        print("Validation accuracy did not improve")

def load_checkpoint(model):
    if os.path.isfile(config.checkpoint_file):
        config.checkpoint_flag = 1
        print("Loading checkpoint..")
        if config.cuda:
            checkpoint = torch.load(config.checkpoint_file)
        else:
            checkpoint = torch.load(config.checkpoint_file,
                                    map_location=lambda storage,
                                    loc:storage)
        config.start_epoch = checkpoint["epoch"]
        config.model_state = checkpoint["state_dict"]
        config.best_accuracy = checkpoint["best_accuracy"]
        print("Loaded checkpoint {}, (trained for {} epochs)".format(config.checkpoint_file, checkpoint["epoch"]))

def checkpoint(model, data_dir="outputs/checkpoint.pt"): # new_model geliyor
    if os.path.isfile(data_dir):
        new_model = torch.load(data_dir)
    return new_model

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(torch.tensor(image, requires_grad=True))
    return image

def rotate(model=None, images=None):
    
    pil_images = transforms.ToPILImage(images)
    tensor_images = transforms.ToTensor(pil_images)
    gray_pil_images = transforms.functional.to_grayscale(pil_images,output_chanels=3)
    crops = transforms.functional.five_crop(img=pil_images, size=224)
    crop_scores = {}
    for i in range(5):
        crop_scores[i] = model(crops)
    print(type(crop_scores), crop_scores.shape)


