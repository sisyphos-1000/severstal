from torchvision import transforms
from data_utils import SteelDataset, tensor_to_set, img_mask_overlay
import torch
from torch.utils.data import DataLoader
import pandas as pd
import unittest

transform = transforms.Compose(
        [transforms.Resize((256,1600)),
         transforms.ToTensor()
         ])
df_pivot = pd.read_csv('train_pivot.csv')

steelset = SteelDataset(root_dir="train_images",df=df_pivot,nr_classes = 4,transform = transform)
dataloader = DataLoader(steelset,batch_size=12,shuffle='True')


#random binary mask
category = 4
nr_categories = 4
height,width = 256,1600
mask = (torch.randn((height,width)) > 0.5).int()*category
mask_one_hot = steelset._load_mask_one_hot(mask=mask,nr_categories=nr_categories)

assert mask_one_hot.shape == (1,nr_categories,height,width),"Error, shape is {}".format(mask_one_hot.shape)
assert tensor_to_set(mask_one_hot[0,category-1,:,:]) == {0,1},"Error: set is {}".format(tensor_to_set(mask_one_hot[0,category-1,:,:]))

imgs,masks,labels = iter(dataloader).next()

print("All Tests Passed!")
img_mask_overlay(imgs,masks,alpha=1)



class TestSteelDataset(unittest.TestCase):

    def test_mask_one_hot(self):
        category = 4
        nr_categories = 4
        height, width = 256, 1600
        mask = (torch.randn((height, width)) > 0.5).int() * category
        mask_one_hot = steelset._load_mask_one_hot(mask=mask, nr_categories=nr_categories)
        assert mask_one_hot.shape == (1, nr_categories, height, width), "Error, shape is {}".format(mask_one_hot.shape)
        assert tensor_to_set(mask_one_hot[0, category - 1, :, :]) == {0, 1}, "Error: set is {}".format(
            tensor_to_set(mask_one_hot[0, category - 1, :, :]))


