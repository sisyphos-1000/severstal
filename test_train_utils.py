import unittest
from train_utils import calc_correct_preds,lr_rule
import torch
from matplotlib import pyplot as plt
import functools as ft


class TestTrainingLoop(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTrainingLoop,self).__init__(*args,**kwargs)
        self.pred_zeros = torch.zeros((4,5,1))
        self.pred_ones = torch.ones((4,5,1))

        self.pred_1 = torch.zeros((4,5,1))
        self.pred_1[:,4,0] = 1

    def test_calc_accuracy(self):
        assert calc_correct_preds(self.pred_zeros,self.pred_zeros) == 1
        assert calc_correct_preds(self.pred_ones,self.pred_ones) == 1
        assert calc_correct_preds(self.pred_ones,self.pred_zeros) == 0
        assert calc_correct_preds(self.pred_ones,self.pred_1) <1

    def test_lr_rule(self):
        epochs = range(0,200)
        lr = []
        start_epoch = 10
        lr_rule_10 = ft.partial(lr_rule,start_epoch=start_epoch)
        for i in epochs:
            lr.append(lr_rule_10(i))
        plt.plot(epochs,lr)
        plt.show()




