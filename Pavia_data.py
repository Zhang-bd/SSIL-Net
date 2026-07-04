import matplotlib.pyplot as plt
from h5py import File
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import torch
import cv2
import random
from scipy import signal


class PaviaDataset(Dataset):
    def __init__(self, type='train', aug=False):
        super(PaviaDataset, self).__init__()

        self.hsi_channel = 93
        self.msi_channel = 4

        # self.patch_h = 6  # 6
        # self.patch_w = 6  # 6
        self.type = type
        self.aug = aug
        # Generate samples and labels
        # if type != 'test':
        #     self.hrhsi, self.lrhsi, self.hrmsi, self.pan = self.getData()

        if self.type == 'train':
            self.hsi_data, self.msi_data, self.pan_data, self.label = self.generateTrain()

        if self.type == 'test':
            self.hsi_data, self.msi_data, self.pan_data, self.label = self.generateTest()

    def generateTrain(self):
        hrhsi = sio.loadmat("/home/s-zhangbd/code/Data/H_M_P/PC4/train/train_data.mat")['gt']
        lrhsi = sio.loadmat("/home/s-zhangbd/code/Data/H_M_P/PC4/train/train_data.mat")['hsi']
        hrmsi = sio.loadmat("/home/s-zhangbd/code/Data/H_M_P/PC4/train/train_data.mat")['msi']
        pan = sio.loadmat("/home/s-zhangbd/code/Data/H_M_P/PC4/train/train_data.mat")['pan']

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)
        if pan.dtype != np.float32: pan = pan.astype(np.float32)

        return lrhsi, hrmsi, pan, hrhsi

    def generateTest(self):
        hrhsi = sio.loadmat("/home/s-zhangbd/code/Data/H_M_P/PC4/test/test_data.mat")['gt']
        lrhsi = sio.loadmat("/home/s-zhangbd/code/Data/H_M_P/PC4/test/test_data.mat")['hsi']
        hrmsi = sio.loadmat("/home/s-zhangbd/code/Data/H_M_P/PC4/test/test_data.mat")['msi']
        pan = sio.loadmat("/home/s-zhangbd/code/Data/H_M_P/PC4/test/test_data.mat")['pan']

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)
        if pan.dtype != np.float32: pan = pan.astype(np.float32)

        return lrhsi, hrmsi, pan, hrhsi

    def __getitem__(self, index):
        hrhsi = self.label[index]
        hrmsi = self.msi_data[index]
        lrhsi = self.hsi_data[index]
        pan = self.pan_data[index]
        if self.aug == True:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hrhsi = np.rot90(hrhsi)
                hrmsi = np.rot90(hrmsi)
                lrhsi = np.rot90(lrhsi)
                pan = np.rot90(pan)

            # Random vertical Flip
            for j in range(vFlip):
                hrhsi = hrhsi[:, ::-1, :].copy()
                hrmsi = hrmsi[:, ::-1, :].copy()
                lrhsi = lrhsi[:, ::-1, :].copy()
                pan = pan[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                hrhsi = hrhsi[::-1, :, :].copy()
                hrmsi = hrmsi[::-1, :, :].copy()
                lrhsi = lrhsi[::-1, :, :].copy()
                pan = pan[::-1, :, :].copy()

        hrhsi = torch.FloatTensor(hrhsi.copy()).permute(2, 0, 1)
        hrmsi = torch.FloatTensor(hrmsi.copy()).permute(2, 0, 1)
        lrhsi = torch.FloatTensor(lrhsi.copy()).permute(2, 0, 1)
        pan = torch.FloatTensor(pan.copy()).permute(2, 0, 1)

        return hrhsi, lrhsi, hrmsi, pan

    def __len__(self):
        return self.label.shape[0]
