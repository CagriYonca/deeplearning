from __future__ import print_function, division
import os
import torch
import glob
import pandas as pd
import numpy as np
import warnings
import transformations as tr

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class NewDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_map, transform=None):
        self.documents_frame = pd.read_csv(csv_file)    # csv dosyasindan bilgileri(yalnizca bilgileri) cek
        self.root_dir = root_dir                        # resimlerin bulundugu klasor
        self.transform = transform                      # resimlere uygulanacak donusumleri belirt
        self.label_map = label_map                      # ToTensor fonksiyonu icin gereken indexleri gir

    def __len__(self):
        return len(self.documents_frame)                # uzunlugu istendiginde csv dosyasindan cekilen bilgileri dondur
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.documents_frame.iloc[idx, 0])   # doc_frame icerisinden resmi ve adini al
        image = io.imread(img_name)                                                 # resmi okuyup image'e at
        documents = self.documents_frame.iloc[idx, 1]                               # etiketleri documents'e at
        sample = {"image" : image, "etiketler": self.label_map[documents]}          # alinan goruntuyu ve etiketin indexini sample'a at

        if self.transform:                                                          # donusum yapilacaksa    
            sample = self.transform(sample)                                         # donusumu uygula
        
        return sample                                                               # o goruntuyu dondur

def prepare_data(csv_1="csv_files/annotations.csv", csv_2="csv_files/images.csv", data_dir="/mnt/data/data/summer_2018/resized_target_files/", batch_size=256, shuffle=None, num_workers=4):
    # csv_annot : [id, image_id, annotation_user_id, annotation_time, annotation_value]
    csv_annots = pd.read_csv(csv_1)
    csv_annots = csv_annots[csv_annots["annotation_user_id"] > 6]
    # csv_images : [id, md5sum, file_name, file_path, source, annotated, reserved_user_id]
    csv_images = pd.read_csv(csv_2)
    image_names = list(map(lambda s: os.path.basename(s), glob.glob(os.path.join(data_dir, "*.jpg"))))
    our_images = csv_images[csv_images["file_name"].isin(image_names)]
    labeled_images = our_images.merge(csv_annots, on="id")[["file_name", "annotation_value"]]
    strip = lambda x : x.lstrip(" u'").rstrip("'")
    labeled_images["annotation_value"] = labeled_images["annotation_value"].map(strip)
    labeled_images = labeled_images[labeled_images["annotation_value"].isin(["receipt", "invoice", "slip", "inforeceipt", "fisandslip"])]
    
    classes = {}
    classes["receipt"] = len(labeled_images[labeled_images["annotation_value"] == "receipt"])
    classes["invoice"] = len(labeled_images[labeled_images["annotation_value"] == "invoice"])
    classes["slip"] = len(labeled_images[labeled_images["annotation_value"] == "slip"])
    classes["inforeceipt"] = len(labeled_images[labeled_images["annotation_value"] == "inforeceipt"])
    classes["fisandslip"] = len(labeled_images[labeled_images["annotation_value"] == "fisandslip"])

    print(classes["receipt"], classes["invoice"], classes["slip"], classes["inforeceipt"], classes["fisandslip"])

    clean_labeled_images = labeled_images
    clean_labeled_images.to_csv("csv_files/loader.csv", index=None)

    img_name = clean_labeled_images.iloc[:, 0]
    img_label = clean_labeled_images.iloc[:, 1]

    label_map = {}
    for i, label in enumerate(pd.read_csv("csv_files/loader.csv")["annotation_value"].unique()):
        label_map[label] = np.array(i)
    
    clean = pd.read_csv("csv_files/loader.csv")
    
    train_size = int(len(clean) * 0.8)
    clean_train = clean[:train_size]

    clean_train.to_csv("csv_files/clean_train.csv", index=False)
    
    test_size = int((len(clean) - train_size) / 2)
    clean_test = clean[train_size:(train_size + test_size)]

    clean_test.to_csv("csv_files/clean_test.csv", index=False)
    val_size = len(clean) - train_size - test_size

    clean_val = clean[(train_size + test_size):]
    clean_val.to_csv("csv_files/clean_val.csv", index=False)

    new_transforms = [
    tr.RandomCrop(224),
    tr.ToTensor(),
    ]

    sample_train = NewDataset("csv_files/clean_train.csv", data_dir, label_map=label_map, transform=transforms.Compose(new_transforms))


    train_loader = DataLoader(sample_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    sample_test = NewDataset("csv_files/clean_test.csv", data_dir, label_map=label_map, transform=transforms.Compose([
    tr.RandomCrop(224), tr.ToTensor(),
    ]))


    test_loader = DataLoader(sample_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    sample_val = NewDataset("csv_files/clean_val.csv", data_dir, label_map=label_map, transform=transforms.Compose([
    tr.RandomCrop(224), tr.ToTensor(),
    ]))


    val_loader = DataLoader(sample_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


    samples = {"train" : train_loader, "test" : test_loader, "val" : val_loader}
    dataset_sizes = {"train" : train_size, "test" : test_size, "val" : val_size}

    return (samples, dataset_sizes, label_map)
