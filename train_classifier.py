import wandb
from torch import nn,optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
import pandas as pd
import train_utils
import data_utils as utils
import torch
import numpy as np
import os
import functools as ft


seed = 23
torch.manual_seed(seed)  # pytorch random seed


resnet = models.quantization.resnet18(pretrained = True)


#Loss function will handle Sigmoid in output
classification_head = nn.Sequential(nn.Flatten(),
                                    nn.Linear(1024*3,512),
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


        #return {'loss': loss,}


transform = transforms.Compose(
        [transforms.Resize((32,32)),
         transforms.ToTensor()
         ])



df_pivot = pd.read_csv('train_pivot.csv')

df_test_small = df_pivot[0:40]

trainset_classification = utils.SteelDataset(root_dir="train_images_small",df=df_test_small,nr_classes = 4,transform = transform)
train_loader = DataLoader(trainset_classification,batch_size=20,shuffle='True')

valset_classification = utils.SteelDataset(root_dir="train_images_small",df=df_test_small,nr_classes = 4, transform = transform)
val_loader = DataLoader(trainset_classification,batch_size=20,shuffle='True')



wandb_logger = WandbLogger(project='severstal')

trainer = pl.Trainer(max_epochs=200,log_every_n_steps=1,logger=wandb_logger)
model = LitResNet(classification_head)



trainer.fit(model,train_loader,val_loader)


##############################################################################################
#setup logging and config
'''
project_name = 'severstal_classification'

config = wandb.config

config.batch_size = 1
config.val_batch_size = 4
config.nr_epochs = 200
config.learning_rate = 1e-4
config.device = 'cpu'
config.seed = 42
config.log_interval = 10
config.criterion = nn.BCELoss()
config.optimizer = optimizer
config.begin_decay = 100
#config.lr_schedule = lr_rule
config.model = resnet
config.transform = transform

wandb.init(
  project=project_name,
  config=config,
)

wandb_api = wandb.Api()
run_name = wandb_api.runs(path='sisyphos/'+project_name)[0].name
os.makedirs(run_name,exist_ok=True)

wandb.watch(resnet)



####################################################
# misc parameters that are not logged

start_epoch = 0
val_each_epoch = 2
save_each_epoch = 10

#lr_scheduler = torch.optim.lr_scheduler.LambdaLR(config.optimizer, lr_lambda=lr_rule)






########################################################################################
#training


for epoch in range(start_epoch,config.nr_epochs):
    print("Epoch {}: Training".format(epoch))
    config.learning_rate = config.lr_schedule(config.learning_rate,epoch)
    losses, correct_preds = train_utils.train(resnet,config.device,trainloader_classification,optimizer,config.criterion,epoch,wandb)
    mean_train_loss = np.mean(losses)
    mean_train_accuracy = np.mean(correct_preds)
    wandb.log({"train_loss": mean_train_loss, "train_accuracy_epoch": mean_train_accuracy})
    wandb.log({"learning_rate":config.learning_rate})
    lr_scheduler.step(epoch)
    if epoch%save_each_epoch == 0:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': resnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler,
                    }, os.path.join('runs',run_name,'resnet15_epoch_{}.pth'.format(epoch)))


    if epoch%val_each_epoch == 0 and epoch > 0:
        print("Epoch {}: Validating".format(epoch))
        val_loss, val_preds = train_utils.validate(resnet,config.device,valloader_classification,config.criterion,epoch,wandb)
        mean_val_loss = np.mean(val_loss)
        mean_val_accuracy = np.mean(val_preds)
        wandb.log({"val_loss": mean_val_loss, "val_accuracy": mean_val_accuracy})

'''
