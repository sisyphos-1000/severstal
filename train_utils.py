import torch
from tqdm import tqdm
import time

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


def dummy_train(trainloader):

    for img,mask,labels in tqdm(trainloader):
        pass



def train(model, device, trainloader, optimizer, criterion, epoch, wandb=None):
    '''
    Training Routine for a single epoch.

    Params
    ------
    model:
    device:
    trainloader:
    optimizer:
    criterion:
    epoch: necessary for learning rate schedule
    steps_per_epoch:
    logs:

    '''
    model = model.to(device)
    model.train()
    losses = []
    correct_preds = []

    for imgs, _, labels in tqdm(trainloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred = model(imgs)

        loss = criterion(pred, labels)
        correct = calc_correct_preds(pred,labels)
        correct_preds.extend(correct.flatten().tolist())

        loss.backward()
        losses.append(loss.item())

        if epoch > 5:
            pass


        optimizer.step()

    return losses,correct_preds



    return model

def validate(model,device,valloader,criterion,epoch,wandb=None):
    '''

    :param model: model to validate
    :param device:
    :param valloader:
    :param optimizer:
    :param criterion:
    :param epoch: current epoch
    :param wandb: logging object
    :return:
    '''
    model = model.to(device)
    model.eval()
    losses = []
    correct_preds = []
    with torch.no_grad():
        for imgs, _, labels in tqdm(valloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            pred = model(imgs)

            loss = criterion(pred, labels)
            correct = calc_correct_preds(pred,labels)
            correct_preds.extend(correct.flatten().tolist())

            losses.append(loss)

    return losses, correct_preds




def lr_rule(epoch,start_epoch):
    # learning rate goes linearly to zero starting from epoch start_epoch

    return 1.0 - max(0, epoch + 1 - start_epoch) / start_epoch






