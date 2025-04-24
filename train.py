from torch.utils.data import DataLoader
import torch.nn as nn
from Model.m import Model
import os
import torch
import numpy as np
from loss import cal_sam
from Chi_hmp import ChikuseiDataset
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import torch.nn.init as init
from tqdm import tqdm
from psnr import MPSNR

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
def set_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    



if __name__ == '__main__':
    set_seed(34)
    epochs = 1000
    bs = 4
    lr = 1e-4
    train_data = ChikuseiDataset(type='train',aug=True)
    train_loader = DataLoader(dataset=train_data, num_workers=10, batch_size=bs, shuffle=True,
                              pin_memory=True, drop_last=False)
    model = Model(128, 4, 1).cuda()
        
    
    loss_func = nn.L1Loss().cuda()
    #amploss = AMPLoss().cuda()
    #phaloss = PhaLoss().cuda()

    path = 'hmp_check/m'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Path '{path}' created.")
    else:
        print(f"Path '{path}' already exists.")

    log_file_path = os.path.join(path, 'training_log.txt')
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    #scheduler = StepLR(optimizer=optimizer, step_size=100, gamma=0.1)
    #scheduler = MultiStepLR(optimizer, milestones=[100, 150, 175, 190, 195], gamma=0.5)
    scheduler = MultiStepLR(optimizer, milestones=list(range(100, 1000, 25)), gamma=0.5)
    
    test_data = ChikuseiDataset(type='test',aug=False)
    val_loader =  DataLoader(dataset=test_data, num_workers=10, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=False)

    best_loss = 100
    best_psnr = 0
    for epoch in range(epochs+1): 
        model.train()
        loss_list = []
        for idx,batch in enumerate(train_loader):
            ref,hsi,msi,pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            out = model(pan,msi,hsi)
            loss = loss_func(ref,out) + 0.1*(cal_sam(out,ref)) #+ 0.01*amploss(out,ref) + 0.01 * phaloss(out,ref)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()
        print(f'Epoch:{epoch},loss:{np.mean(loss_list)}')
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(path, 'model_epoch_{}.pth'.format(epoch)))
        if epoch % 10 == 0 :
            model.eval()
            val_loss_list = []
            psnr_list = []
            with torch.no_grad():
                for val_batch in val_loader:
                    ref_val, hsi_val, msi_val, pan_val = val_batch[0].cuda(), val_batch[1].cuda(), val_batch[2].cuda(), val_batch[3].cuda()
                    out_val = model(pan_val, msi_val, hsi_val)
                    
                    HX = torch.squeeze(out_val).permute(1, 2, 0).cpu().numpy()
                    X = torch.squeeze(ref_val).permute(1, 2, 0).cpu().numpy()
                    psnr = MPSNR(HX,X)
                    psnr_list.append(psnr)
                    
                    val_loss = loss_func(ref_val, out_val)
                    val_loss_list.append(val_loss.item())
            print(f"Epoch: {epoch}, Train Loss: {np.mean(loss_list)}, Validation Loss: {np.mean(val_loss_list)}, avg PSNR: {np.mean(psnr_list)}.")  
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Epoch: {epoch}, Train Loss: {np.mean(loss_list)}, Validation Loss: {np.mean(val_loss_list)}, avg PSNR: {np.mean(psnr_list)}.\n")
            if np.mean(val_loss_list) < best_loss:
                best_loss = np.mean(val_loss_list)
                torch.save(model.state_dict(), os.path.join(path, 'model_best_loss.pth'))
                
            if np.mean(psnr_list) > best_psnr:
                best_psnr = np.mean(psnr_list)
                torch.save(model.state_dict(), os.path.join(path, 'model_best_psnr.pth'))
  
