import torch
import os

cuda = torch.cuda.is_available()

training_batch_size = 64
training_save_dir = "outputs/resnet18-1.pt"
#training_load_dir = "outputs/densenet161new.pt"
training_load_dir = None
training_fc_flag = 1
training_rotate_flag = True
training_num_epochs = 250
training_lr = 0.001
tester_batch_size = 64
tester_load_dir = "outputs/resnet18new.pt"
rotate_flag = True
start_epoch = 0
best_accuracy = 0
model_state = ""
directory = "/mnt/data/data/summer_2018/resized_target_files/"
checkpoint_flag = 0
checkpoint_file = "outputs/checkpoint.pt"

if cuda:
	device = torch.device("cuda")
	print(device)
