from torch import nn,optim
from torchvision import models, transforms
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pandas as pd
import data_utils as utils
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import segmentation_models_pytorch as smp
from torchmetrics import IoU
import albumentations as A
from torchvision.transforms import InterpolationMode
import cv2
import os
import glob
from PIL import Image
from lovasz_loss import lovasz_softmax


seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def calc_correct_preds(pred,labels):
    '''
    calculate accuracy between 0 and 1 for single batch.
    results of this function are added up in each step, to form the
    overall accuracy per epoch

    ONlY WORKS IF BATCHSIZE IS ALWAYS THE SAME (drop incomplete batches!)

    :param pred:
    :param labels:
    :return: accuracy between 0 and 1
    '''
    result = (torch.where(pred > 0.5, 1, 0) == labels).all(axis=1).float()

    return result.mean(axis=0)

def one_hot2mask(mask_one_hot):
    '''
    :param mask_one_hot: shape (B,C,H,W) between [0,1]
    :return mask: shape (B,H,W) between [0,C]
    '''
    B,C,H,W = mask_one_hot.shape
    mask = torch.zeros((B,H,W))
    for index in range(C):
        mask = mask_one_hot[:,index,:,:].squeeze()*index
    return mask



class LitUnet(pl.LightningModule):
    def __init__(self,unet,nr_classes):
        super().__init__()
        self.unet = unet('resnet34',in_channels=3,classes = nr_classes)
        #self.criterion = lovasz_softmax
        self.criterion = nn.BCEWithLogitsLoss()
        self.calc_IoU = IoU(num_classes=nr_classes)
    def forward(self,x):
        return self.unet(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'mean_val_iou'}


    def training_step(self,batch,batch_idx):
        imgs, masks,_ = batch


        pred = self(imgs)

        loss =  self.criterion(pred,masks.float())
        pred_thres = (pred > 0.5).int()

        mask_cat = one_hot2mask(masks.detach().int())

        iou = self.calc_IoU(pred_thres.detach(),masks.detach())

        self.log('train_loss',loss)
        self.log('train_iou',iou)

        return {'loss':loss, 'iou':iou}


    def validation_step(self,batch,batch_idx):
        results = self.training_step(batch,batch_idx)
        return results


    def validation_epoch_end(self,val_step_outputs):
        mean_val_loss = torch.Tensor([i['loss'] for i in val_step_outputs]).mean()
        mean_val_iou = torch.Tensor([i['iou'] for i in val_step_outputs]).mean()

        self.log('mean_val_loss',mean_val_loss)
        self.log('mean_val_iou',mean_val_iou)

        return {'val_loss': mean_val_loss, 'val_iou':mean_val_iou}




df = pd.read_csv('train.csv')

df_pivot = utils.df_steel_to_pivot(df)

#df_pivot_dummy = df_to_pivot(df[0:20])



transform = transforms.Compose(
        [transforms.Resize((128,128)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225])
         ])

masktransform = transforms.Compose(
        [transforms.Resize((128,128),InterpolationMode.NEAREST),
         transforms.ToTensor()
         ])


augmentation_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate( border_mode=cv2.BORDER_CONSTANT,
                          shift_limit_x=[-0.25,0.25],
                          shift_limit_y=0,
                          rotate_limit = 0,
                          scale_limit = 0,
                          p=0.5)
], p=1)



#segmentation datasets only contain positive samples
df_pivot_segmentation = df_pivot.loc[df_pivot.iloc[:,1:].any(axis=1),:] #all defects

#validation set is deliberately much smaller. in segmentation, one sample gives more info
df_train_segmentation,df_val_segmentation = train_test_split(df_pivot,train_size=0.95,test_size=0.05)



trainset_segmentation = utils.SteelDataset(root_dir='train_images',
                                             df=df_train_segmentation,
                                             nr_classes = 4,
                                             transform = transform,
                                             mask_transform = masktransform,
                                             aug_transform = augmentation_transform)

train_loader_segmentation = DataLoader(trainset_segmentation,batch_size=5,shuffle='True',num_workers=2)

valset_segmentation = utils.SteelDataset(root_dir='train_images',
                                           df=df_val_segmentation,
                                           nr_classes = 4,
                                           transform = transform,
                                           mask_transform = masktransform,
                                           aug_transform = None)
val_loader_segmentation = DataLoader(valset_segmentation,batch_size=5,shuffle='False',num_workers=2)



checkpoint_callback = ModelCheckpoint(dirpath='lightning_unet_augmentation_bce',every_n_epochs = 1,filename='unet_resnet34-{epoch:02d}-{val_loss:.2f}')

wandb_logger = WandbLogger(project='severstal_segmentation')
trainer_segmentation = pl.Trainer(max_epochs=100,log_every_n_steps=1,
                     check_val_every_n_epoch=1,
                     logger=wandb_logger,
                     #gpus=1,
                     callbacks = [checkpoint_callback],
                     fast_dev_run=False)


#unet will have 7 inputs, 3 for the image, and 4 for the respective class predictions of the input classifier
unet = smp.Unet #Pretrained Unet with resnet encoder
#unet = nn.Sequential(unet,nn.Sigmoid())
model_segmentation = LitUnet(unet,nr_classes=4)



trainer_segmentation.fit(model_segmentation,train_loader_segmentation,val_loader_segmentation)