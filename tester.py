from __future__ import print_function, division
from PIL import ImageFile
from torchvision import datasets, models, transforms

import torch.nn as nn
import warnings
import mydataset
import config
import torch.optim as optim
import functions
import numpy as np
import torch
import model

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")
print("Loading data..")
new_data = mydataset.prepare_data(csv_1="csv_files/annotations.csv", csv_2="csv_files/images.csv", data_dir=config.directory, batch_size=config.tester_batch_size, shuffle=True, num_workers=4)
print("Loading checkpoint..")
new_model = torch.load(config.tester_load_dir)
new_model = new_model.cuda()
print("Loading criterion..")
new_criterion = nn.CrossEntropyLoss().to(config.device)
print("Evaluating model..")
test_ft = model.test_model(
new_model, criterion=new_criterion, 
dataloader=new_data[0], datasizes=new_data[1], 
labels_for_confusion=new_data[2], rotate_flag=config.rotate_flag)
