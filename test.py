from torch.utils.data import DataLoader
import torch.nn as nn
from Model.m import Model
import os
import torch
import numpy as np
from Chi_hmp import ChikuseiDataset
from WDC_data import WdcDataset
from Pavia_data import PaviaDataset
from Real_data2 import RealDataset
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import scipy.io as sio

if __name__ == '__main__':
    #model = Model(93, 4, 1).cuda()#,image_size=256
    model = Model(128, 4, 1).cuda()
    file_path = 'hmp_check2/m/model_best_psnr.pth'#model_best_psnr.pth'
    model.load_state_dict(torch.load(file_path))

    test_data = ChikuseiDataset(type='test', aug=False)
    test_loader = DataLoader(dataset=test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=10
                             )
    # output = np.zeros((3, 256, 256, 93), dtype=np.float32)
    output = np.zeros((12, 256, 256, 128), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for idx,batch in enumerate(test_loader):
            ref,hsi,msi,pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            out = model(pan,msi,hsi)
            output[idx,:,:,:] =  out.permute(0, 2, 3, 1).cpu().detach().numpy()

    sio.savemat('result/m_out.mat',{'output':output})
