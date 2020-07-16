import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import sklearn.metrics as metrics
import cv2
import os
import numpy as np
import pandas as pd

from time import time
from PIL import Image

import warnings

warnings.filterwarnings('ignore')

from networks.trianglenet import TriangleNet
import Constants

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCHSIZE_PER_CARD = 8


def calculate_auc_test(prediction, label):
    # read images
    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()
    # label_1D = label_1D / 255

    if len(np.unique(label_1D)) != 2:
        return None

    auc = metrics.roc_auc_score(label_1D, result_1D)
    # print("AUC={0:.4f}".format(auc))
    return auc

def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    return acc

def sensitivity(pred_mask, label):
    '''
    sen=TP/(TP + FN)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN = [0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
    if TP + FN != 0:
        sen = TP / (TP + FN)
        return sen
    else:
        return None

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))

        # process the input image to fix the batch size
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        content = mask[:, :, :448]
        contour = mask[:, :, 448:]
        masks = []
        for m in [content, contour]:
            mask1 = m[:4] + m[4:, :, ::-1]
            mask2 = mask1[:2] + mask1[2:, ::-1]
            mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
            masks.append(mask3)

        return np.concatenate(masks, axis=1)

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)


def test_triangle_net(sources, prefix='TCGA', target='logs/log_TriangleNet_Gauss', csv_target='logs/predict_trianglenet_gauss.csv'):
    val = [(fn, source) for source in sources for fn in os.listdir(os.path.join(source, 'images')) if fn[:len(prefix)] == prefix]
    disc = 20
    solver = TTAFrame(TriangleNet)
    solver.load('weights/TriangleNet_Gauss.th')

    if not os.path.exists(target):
        os.mkdir(target)
    
    total_acc = []
    total_sen = []
    threshold = 4
    total_auc = []

    table = {
        'image': [],
        'acc': [],
        'sen': [],
        'auc': []
    }

    for i, t in enumerate(val):
        name, source = t
        image_path = os.path.join(source, 'images', name)
        print(image_path)
        gt_root = os.path.join(source, 'masks')
        
        mask = solver.test_one_img_from_path(image_path)

        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0

        content_path = os.path.join(gt_root, name[:23] + '_content' + name[23:])
        content = cv2.imread(content_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, dsize=(np.shape(content)[1] * 2, np.shape(content)[0]))
        mask = mask[:, :content.shape[1]]

        predi_mask = np.zeros(shape=np.shape(mask))
        predi_mask[mask > disc] = 1
        gt = np.zeros(shape=np.shape(content))
        gt[content > 0] = 1
        pm = predi_mask

        acc = accuracy(pm, gt)
        sen = sensitivity(pm, gt)
        auc = calculate_auc_test(pm, gt)
        total_acc.append(acc)
        if not sen is None:
            total_sen.append(sen)
        if not auc is None:
            total_auc.append(auc)

        print(i + 1, 'acc:', acc, 'sen:', sen, 'auc:', auc)
        table['image'].append(name)
        table['acc'].append(acc)
        table['sen'].append(sen)
        table['auc'].append(auc)

        # cv2.imwrite(os.path.join(target, name[:23] + '_predict' + name[23:]), mask.astype(np.uint8))

    print(np.mean(total_acc), np.std(total_acc))
    print(np.mean(total_sen), np.std(total_sen))
    print(np.mean(total_auc), np.std(total_auc))
    
    dft = pd.DataFrame(table)
    dft.to_csv(csv_target, index=0)

if __name__ == '__main__':
    test_triangle_net([os.path.join(Constants.ROOT, 'test')])
"""
    sources = [os.path.join(Constants.ROOT, 'test'), os.path.join(Constants.ROOT, 'train')]
    slides = ['TCGA-DC-6681-01A-01-BS1', 'TCGA-EI-6507-01A-01-BS1', 'TCGA-AA-3684-01A-01-BS1', 'TCGA-A6-6651-01A-01-BS1', 'TCGA-AA-3695-01A-01-BS1', 'TCGA-AG-3893-01A-01-BS1']
    for slide in slides:
        test_triangle_net(sources, slide, f'logs/log_tng_{slide}', f'logs/predict_tng_{slide}.csv')
"""