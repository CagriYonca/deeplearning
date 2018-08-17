from __future__ import print_function, division
from PIL import ImageFile
from torchvision import datasets, models, transforms

import warnings
import mydataset
import model
import config
import torch.optim as optim
import functions
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

new_data = mydataset.prepare_data(csv_1="csv_files/annotations.csv", csv_2="csv_files/images.csv", data_dir=config.directory, batch_size=config.training_batch_size, shuffle=True, num_workers=4)

new_model = models.resnet18(pretrained=True)

new_optimizer = optim.Adam(new_model.parameters(), lr=config.training_lr)
#new_optimizer = optim.SGD(new_model.parameters(), lr=config.training_lr, momentum=0.9)

model_ft, criterion, optimizer_ft, exp_lr_scheduler = model.build_model(model_name=new_model, model_path=config.training_load_dir, fc_flag=config.training_fc_flag, num_of_feature=5, optimizer=new_optimizer)

model_ft = model.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=config.training_num_epochs, dataloader=new_data[0], datasizes=new_data[1], load_dir=config.training_load_dir, save_dir=config.training_save_dir, rotate_flag=config.training_rotate_flag)
    


