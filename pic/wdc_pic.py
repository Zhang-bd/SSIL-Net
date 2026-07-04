import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def show(data,id,name):
    t = np.squeeze(data[id])
    band = (59-1,47-1,40-1)#(35-1, 16-1, 7-1)
    # band = 80
    rgb = t[:, :, band]

    plt.imshow(rgb, cmap='gray')
    # plt.title('GT3')
    plt.axis('off')
    save_path = r'H:\work2\new_hsi_msi_pan\pic_false_color\WDC\pic2\0\\{}.png'.format(name)

    plt.imsave(save_path, rgb, cmap='gray')
    # plt.show()

def err_map(gt,data,id,name,band):
    output = np.squeeze(data[id])
    img_refference = np.squeeze(gt[id])
    plt.imshow(abs(output[:, :, band] - img_refference[:, :, band]), vmin=0, vmax=0.20, cmap='jet') # 0.25
    plt.axis('off')
    save_path = r'H:\work2\new_hsi_msi_pan\pic_false_color\WDC\error2\0\\{}.png'.format(name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()
    return



if __name__ == '__main__':

    gt = sio.loadmat(r"../wangstondc/data/test_gt.mat")['gt']  # DC
    # print(im_gt.shape) # (2, 304, 304, 191)

    hyper = sio.loadmat(r"H:\work2\结果\Hyper\hyper2\hyper_DC_304.mat")['output']  # hypernet new
    # our = sio.loadmat(r"H:\work2\结果\Ours_dc\new_model8_13_dc_1.mat")['output']
    our = sio.loadmat(r"H:\work2\结果\Ours_dc\m2\m2_wdc_psnr.mat")['output']

    hmp = sio.loadmat(r"H:\work2\结果\HMPNet\WDC\out_dc_2_304_1.mat")['output'] # HMP
    psrt = sio.loadmat(r"H:\work2\结果\PSRT\psrt_dc_820.mat")['output'] # psrt
    scp = sio.loadmat(r"H:\work2\结果\SCPNet\out\scp_wdc.mat")['output']
    ssr = sio.loadmat(r"H:\work2\结果\SSRnet\ssr_dc_best.mat")['output']  # DC ssrnet
    hsr = sio.loadmat(r"H:\work2\结果\Hsr\hsr_dc_best.mat")['output']  # DC hsrnet
    dsp = sio.loadmat(r"H:\work2\结果\DSP\dsp_dc_best.mat")['output']  # DC dspnet
    mimo = sio.loadmat(r"H:\work2\结果\MIMO\MIMO_hmp_dc_best.mat")['output']  # DC mimo-sst
    dt3 = sio.loadmat(r"H:\work2\结果\3DT\3dt_hmp_dc_100.mat")['output']  # DC 3dt
    dpf = sio.loadmat(r"H:\work2\结果\DPFormer\wdc_1.mat")['output']
    dim = sio.loadmat(r"H:\work2\结果\DIM\wdc_2.mat")['output']
    pmi = sio.loadmat(r"H:\work2\结果\PMII\result_hmp\PMI_wdc_1.mat")['output']  # PMI real

    # index = ['ssr','hsr','scp','hyper','hmp','dsp','psrt','mimo','dt3','pmi','our','gt'] #['ssr','hsr','hyper','hmp','dsp','psrt','mimo','dt3','pmi','our']
    index = ['dpf','dim']
    # 生成伪彩色图像
    for i in index:
        print("----processing----",i)
        im_out =  globals()[i]
        print(np.max(im_out),np.min(im_out))
        im_out = np.clip(im_out,0,1)
        print(np.max(im_out), np.min(im_out))

        show(im_out,id=0,name=i)
    # 生成残差图像
    for i in index:
        print("----processing----",i)
        im_out =  globals()[i]
        print(np.max(im_out),np.min(im_out))
        im_out = np.clip(im_out,0,1)
        print(np.max(im_out), np.min(im_out))

        err_map(gt=gt,data=im_out,id=0,name=i,band=46)

