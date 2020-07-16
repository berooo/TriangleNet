from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

from networks.trianglenet import TriangleNet
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
from Visualizer import Visualizer

import Constants
import image_utils

# Please specify the ID of graphics cards that you want to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def TriangleNet_Train():
    NAME = 'TriangleNet'

    # run the Visdom
    viz = Visualizer(env=NAME)

    solver = MyFrame(TriangleNet, dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD

    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    dataset = ImageFolder(root_path=Constants.ROOT, datasets='content_contour')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)

    # start the logging files
    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()

    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0

        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss
            index = index + 1

        # show the original images, predication and ground truth on the visdom.
        show_image = (img + 1.6) / 3.2 * 255.
        viz.img(name='images', img_=show_image[0, :, :, :])
        viz.img(name='labels', img_=mask[0, :, :, :])
        viz.img(name='prediction', img_=pred[0, :, :, :])

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        print('********', file=mylog)
        print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
        print('train_loss:', train_epoch_loss, file=mylog)
        print('SHAPE:', Constants.Image_size, file=mylog)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', Constants.Image_size)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '.th')
        if no_optim > Constants.NUM_EARLY_STOP:
            print('early stop at %d epoch' % epoch, file=mylog)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + NAME + '.th')
            solver.update_lr(2.0, factor=True, mylog=mylog)
        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    TriangleNet_Train()



