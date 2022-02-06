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





def lr_rule(epoch,start_epoch):
    # learning rate goes linearly to zero starting from epoch start_epoch

    return 1.0 - max(0, epoch + 1 - start_epoch) / start_epoch






