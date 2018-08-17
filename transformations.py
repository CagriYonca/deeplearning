import os
import torch
import glob
import pandas as pd
import numpy as np
import warnings
import random

from skimage import io, transform, color
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Rescale(object):                                                              # rescale edici nesne olustur
    def __init__(self, output_size):                                                # baslatildiginda
        assert isinstance(output_size, (int, tuple))                                # int veya tuple olup olmadigini kontrol et
        self.output_size = output_size                                              # int veya tuple ise output size olarak ata

    def __call__(self, sample):                                                     # cagirildiginda
        image, documents = sample["image"], sample["etiketler"]                     # image ve documents adinda resmi ve etiketi al
        h, w = image.shape[:2]                                                      # eni ve boyu al
        if isinstance(self.output_size, int):                                       # output size int ise
            if h > w:                                                               # boyu eninden buyukse
                new_h, new_w = self.output_size * h / w, self.output_size           # boyu oraniyla degistirip ene size'i direkt uygula
            else:           
                new_h, new_w = self.output_size, self.output_size * w / h           # eni ortaniyla degistirip boya size'i direkt uygula
        else:
            new_h, new_w = self.output_size                                         # tuple ise en ve boyu direkt uygula

        new_h, new_w = int(new_h), int(new_w)                                       # tam degere ata
        img = transform.resize(image, (new_h, new_w))                               # resmi olcekle

        return {"image": img, "etiketler": documents}

class Grayscale(object):
    def __init__(self):
        self.i = 0
    def __call__(self, sample):
        image, documents = sample["image"], sample["etiketler"]
        self.i += 1
        image = color.rgb2gray(image)
        image = np.append(image.shape, 1)

        return {"image": image, "etiketler": documents}

class RandomRotate(object):
    def __init__(self):
        self.i = 0
    def __call__(self, sample):
        image, documents = sample["image"], sample["etiketler"]
        image = transform.rotate(image, 90)
        return {"image" : image, "etiketler" : documents}

class RandomCrop(object):                                   # rastgele crop alma nesnesi
    def __init__(self, output_size):                        # baslatilirken
        assert isinstance(output_size, (int, tuple))        # size int veya tuple m
        if isinstance(output_size, int):                    # int ise
            self.output_size = (output_size, output_size)   # tuple olarak al(cikis kare olmasi icin)
        else:
            assert len(output_size) == 2                    # tuple ise, 2 veri varsa
            self.output_size = output_size                  # cikis verisi olarak bu tuple al

    def __call__(self, sample):                                 # bir ornek ile cagirildiginda
        image, documents = sample["image"], sample["etiketler"] # ornegin resim ve etiketler verisini ayristir
        h, w = image.shape[:2]                                  # resmin en ve boyunu al
        new_h, new_w = self.output_size                         # yeni en ve boyu al

        top = np.random.randint(0, h - new_h)                   # 0 ile tum boyut ile crop boyutu farkindan topu rastgele al
        left = np.random.randint(0, w - new_w)                  # 0 ilse tum boyut ile crop boyutu farkindan lefti rastgele al
        image = image[top: top + new_h, left: left + new_w]     # resmi cropla
        return {"image": image, "etiketler": documents}          # cropu ve etiketleri dondur

class ToTensor(object):                                                                         # Tensore donusturucu
    def __call__(self, sample):                                                                 # baslatildiginda
        image, documents = sample["image"], sample["etiketler"]                                 # cagrilirken kullanilan sample'in resim ve eti
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        if image.shape[2] == 3:
            image = image.transpose((2, 0, 1))                                                      # ketini al, traspose'unu al
        return {"image" : torch.from_numpy(image), "etiketler" : torch.from_numpy(documents)}    # gelen verileri numpy array'den tensore cevirip ata

