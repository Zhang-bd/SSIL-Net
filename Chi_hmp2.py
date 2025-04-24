import matplotlib.pyplot as plt
from h5py import File
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import torch
import cv2
import random
from scipy import signal

def Gaussian_downsample(x,psf,s):
    x = np.transpose(x, (2, 0, 1))
    y=np.zeros((x.shape[0],int(x.shape[1]/s),int(x.shape[2]/s)))
    if x.ndim==2:
        x=np.expand_dims(x,axis=0)
    for i in range(x.shape[0]):
        x1=x[i,:,:]
        x2=signal.convolve2d(x1,psf, boundary='symm',mode='same')
        y[i,:,:]=x2[0::s,0::s]
    y = np.transpose(y, (1,2,0))
    return y

class ChikuseiDataset(Dataset):
    def __init__(self, h_stride=4, w_stride=4, type='train', aug=False):
        super(ChikuseiDataset, self).__init__()
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.hsi_channel = 128
        self.msi_channel = 4

        self.patch_h = 6 #6
        self.patch_w = 6 #6
        self.type = type
        self.aug = aug
        # Generate samples and labels
        # if type != 'test':
        #     self.hrhsi, self.lrhsi, self.hrmsi, self.pan = self.getData()

        if self.type == 'train':
            self.hsi_data, self.msi_data, self.pan_data, self.label = self.generateTrain()

        if self.type == 'test':
            self.hsi_data, self.msi_data, self.pan_data, self.label = self.generateTest()

    def getData(self):
        # srf = sio.loadmat('/public/home/s-zhangbd/code/Data/Chikusei/chikusei_128_4.mat')['R']
        srf = np.load('/public/home/s-zhangbd/code/Data/H_M_P/srf/chi_128_4_norm.npy')
        psf_1 = np.load('/public/home/s-zhangbd/code/Data/H_M_P/psf/psf_8_2.npy')
        # mat_save_path = '/public/home/s-zhangbd/code/Data/Chikusei/chikusei_128_4.mat'
        #mat_save_path = r'/public/home/s-zhangbd/code/Data/Chikusei/HMP/Chikusei01.mat'
        #hrhsi = np.array(File(mat_save_path)['chikusei']).transpose([1, 2, 0])
        hrhsi = np.load(r'/public/home/s-zhangbd/code/Data/Chikusei/HMP/Chikusei01.npy')
        #hrhsi /= hrhsi.max()
        hrhsi[300:812, 300:812] = hrhsi[812:1324, 812:1324]
        hrhsi[300:812, 1000:1512] = hrhsi[812:1324, 1512:2024]
        hrhsi[900:1412, 300:812] = hrhsi[1412:1924, 812:1324]
        # hrhsi = np.random.rand(2335, 2517, 128)

        # (2335, 2517, 128)
        # hrhsi = normalize(hrhsi)
        
        #delet black
        # hrhsi = hrhsi[100:-100,100:-100,:]
        hrhsi = hrhsi[70: -73, 70: -79,:]

        #  Generate LRHSI
        # lrhsi = cv2.GaussianBlur(hrhsi, ksize=[7] * 2, sigmaX=2, sigmaY=2)[self.ratio // 2::self.ratio,
        #         self.ratio // 2::self.ratio]
        # #  Generate HRMSI
        hrmsi = hrhsi @ srf
        hrmsi = Gaussian_downsample(hrmsi, psf_1, 4)

        pan = np.mean(hrhsi, axis=2, keepdims=True)

        lrhsi = Gaussian_downsample(hrhsi, psf_1, 16)

        return hrhsi, lrhsi, hrmsi, pan

    def generateTrain(self):
        # patch_h, patch_w = 6, 6
        # rows, cols = self.lrhsi.shape[0], self.lrhsi.shape[1]
        # num = len(list(range(0, rows - patch_h, self.h_stride))) * len(
        #     list(range(0, cols - patch_w, self.w_stride)))
        # # print(num)
        # label_patch = np.zeros((num, patch_h * 16, patch_w * 16, self.hsi_channel), dtype=np.float32)
        # hrmsi_patch = np.zeros((num, patch_h * 4, patch_w * 4, self.msi_channel), dtype=np.float32)
        # lrhsi_patch = np.zeros((num, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        # pan_patch = np.zeros((num, patch_h * 16, patch_w * 16, 1), dtype=np.float32)
        # count = 0

        # # hrhsi, lrhsi, hrmsi = self.getData()
        # hrhsi = self.hrhsi
        # lrhsi = self.lrhsi
        # hrmsi = self.hrmsi
        # pan = self.pan
        # # Data type conversion
        # if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        # if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        # if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)
        # if pan.dtype != np.float32: pan = pan.astype(np.float32)

        # for x in range(0, rows - patch_h, self.h_stride):
        #     for y in range(0, cols - patch_w, self.w_stride):
        #         label_patch[count] = hrhsi[x * 16:(x + patch_h) * 16, y * 16:(y + patch_w) * 16, :]
        #         hrmsi_patch[count] = hrmsi[x * 4:(x + patch_h) * 4, y * 4:(y + patch_w) * 4, :]
        #         lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
        #         pan_patch[count] = pan[x * 16:(x + patch_h) * 16, y * 16:(y + patch_w) * 16, :]
        #         count += 1
        # sio.loadmat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/train/train_data.mat",{"hsi":lrhsi_patch,"msi":hrmsi_patch,"pan":pan_patch,"gt":label_patch})
        lrhsi_patch = sio.loadmat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/train/train_data.mat")['hsi']
        hrmsi_patch = sio.loadmat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/train/train_data.mat")['msi']
        pan_patch = sio.loadmat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/train/train_data.mat")['pan']
        label_patch = sio.loadmat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/train/train_data.mat")['gt']
        return lrhsi_patch, hrmsi_patch, pan_patch, label_patch

    def generateTest(self):
        # test = np.load(r'/public/home/s-zhangbd/code/Data/Chikusei/HMP/test.npy')
        # srf = np.load('/public/home/s-zhangbd/code/Data/H_M_P/srf/chi_128_4_norm.npy')
        # psf_1 = np.load('/public/home/s-zhangbd/code/Data/H_M_P/psf/psf_8_2.npy')
        # num = test.shape[0]
        # patch_h,patch_w = 16,16
        # label_patch = np.zeros((num, patch_h * 16, patch_w * 16, self.hsi_channel), dtype=np.float32)
        # hrmsi_patch = np.zeros((num, patch_h * 4, patch_w * 4, self.msi_channel), dtype=np.float32)
        # lrhsi_patch = np.zeros((num, patch_h, patch_w, self.hsi_channel), dtype=np.float32)
        # pan_patch = np.zeros((num, patch_h * 16, patch_w * 16, 1), dtype=np.float32)
        # for i in range(num):
        #     tt = test[i,:,:,:]
        #     pan_patch[i,:,:,:]  = np.mean(tt, axis=2, keepdims=True)
        #     hrmsi = tt @ srf
        #     hrmsi_patch[i,:,:,:] = Gaussian_downsample(hrmsi, psf_1, 4)
        #     lrhsi_patch[i,:,:,:] = Gaussian_downsample(tt, psf_1, 16)
        #     label_patch[i,:,:,:] = tt

        lrhsi_patch = sio.loadmat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/test/test_data.mat")['hsi']
        hrmsi_patch = sio.loadmat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/test/test_data.mat")['msi']
        pan_patch = sio.loadmat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/test/test_data.mat")['pan']
        label_patch = sio.loadmat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/test/test_data.mat")['gt']


        # Data type conversion
        if label_patch.dtype != np.float32: label_patch = label_patch.astype(np.float32)
        if hrmsi_patch.dtype != np.float32: hrmsi_patch = hrmsi_patch.astype(np.float32)
        if lrhsi_patch.dtype != np.float32: lrhsi_patch = lrhsi_patch.astype(np.float32)
        if pan_patch.dtype != np.float32: pan_patch = pan_patch.astype(np.float32)
        # sio.savemat("/public/home/s-zhangbd/code/Data/H_M_P/Chi2/test/test_data.mat",{"hsi":lrhsi_patch, "msi":hrmsi_patch, "pan":pan_patch, "gt":label_patch})
        return lrhsi_patch, hrmsi_patch, pan_patch, label_patch

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



# if __name__ == "__main__":
#     # train_data = ChikuseiDataset(h_stride=6,w_stride=6,type='train',aug=False)
#     test_data = ChikuseiDataset(type='test',aug=False)
