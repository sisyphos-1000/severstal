import wandb
from torch import nn,optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import train_utils
import data_utils as utils
import torch
from sklearn.model_selection import train_test_split
from torchvision.transforms import InterpolationMode
import albumentations as A

seed = 23
torch.manual_seed(seed)  # pytorch random seed


resnet = models.quantization.resnet18(pretrained = True)




#Loss function will handle Sigmoid in output

classification_head = nn.Sequential(
                                    nn.Linear(1000,512),
                                    #nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Linear(512,64),
                                    #nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Linear(64,4))

resnet = nn.Sequential(resnet,classification_head)


class LitResNet(pl.LightningModule):
    def __init__(self,resnet):
        super().__init__()
        self.resnet = resnet
        self.criterion = nn.BCEWithLogitsLoss()
        self.calc_accuracy = train_utils.calc_correct_preds
    def forward(self,x):
        return self.resnet(x)

    def configure_optimizers(self):
        #optimizer = optim.SGD(resnet.parameters(), lr=1e-1,momentum=0.1)
        optimizer = optim.Adam(resnet.parameters())
        return optimizer

    def training_step(self,batch,batch_idx):
        imgs, _, labels = batch

        pred = self(imgs)

        loss = self.criterion(pred, labels)
        acc = self.calc_accuracy(pred,labels)

        self.log('train_loss',loss)
        self.log('train_acc',acc)
        return {'loss':loss, 'acc':acc}


    def validation_step(self,batch,batch_idx):
        results = self.training_step(batch,batch_idx)
        return results


    def validation_epoch_end(self,val_step_outputs):
        mean_val_loss = torch.Tensor([i['loss'] for i in val_step_outputs]).mean()
        mean_val_acc = torch.Tensor([i['acc'] for i in val_step_outputs]).mean()

        self.log('mean_val_loss',mean_val_loss)
        self.log('mean_val_acc',mean_val_acc)

        return {'val_loss': mean_val_loss, 'acc_loss':mean_val_acc}


transform = transforms.Compose(
        [transforms.Resize((32,100)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225])
         ])

masktransform = transforms.Compose(
        [transforms.Resize((32,100),InterpolationMode.NEAREST),
         transforms.ToTensor()
         ])


augmentation_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5)
], p=1)



df_pivot = pd.read_csv('train_pivot.csv')

df_train,df_val = train_test_split(df_pivot,train_size=0.9,test_size=0.1)

trainset_classification = utils.SteelDataset(root_dir="train_images",df=df_train,nr_classes = 4,transform = transform,mask_transform = masktransform,aug_transform = augmentation_transform)
train_loader = DataLoader(trainset_classification,batch_size=20,shuffle='True')

valset_classification = utils.SteelDataset(root_dir="train_images",df=df_val,nr_classes = 4, mask_transform = masktransform, transform = transform)
val_loader = DataLoader(trainset_classification,batch_size=20,shuffle='True')

assert iter(train_loader).next()[0].shape == (20,3,32,100), f"Error: Img Shape is {iter(train_loader).next()[0].shape}"

assert iter(train_loader).next()[1].shape == (20,4,32,100), f"Error: mask Shape is {iter(train_loader).next()[1].shape}"

wandb_logger = WandbLogger(project='severstal')

trainer = pl.Trainer(max_epochs=100,log_every_n_steps=1,check_val_every_n_epoch=5,logger=wandb_logger)
model = LitResNet(classification_head)



trainer.fit(model,train_loader,val_loader)


##############################################################################################
#setup logging and config
