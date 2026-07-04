import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def show(data,id,name):
    t = np.squeeze(data[id])
    band = (31-1, 16-1, 7-1)
    # band = 80
    rgb = t[:, :, band]

    plt.imshow(rgb, cmap='gray')
    # plt.title('GT3')
    plt.axis('off')
    save_path = './pic/{}_{}.png'.format(id,name)

    plt.imsave(save_path, rgb, cmap='gray')
    # plt.show()

# def err_map(gt,data,id,name,band):
#     output = np.squeeze(data[id])
#     img_refference = np.squeeze(gt[id])
#     plt.imshow(abs(output[:, :, band] - img_refference[:, :, band]), vmin=0, vmax=0.17, cmap='jet')
#     plt.axis('off')
#     save_path = r'H:\work2\new_hsi_msi_pan\pic_false_color\Real\error3\{}.png'.format(name)
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     # plt.show()
#     return

def center_crop(img, target_h, target_w):
    """
    img: (N, H, W, C)
    """
    H, W = img.shape[1], img.shape[2]
    top = (H - target_h) // 2
    left = (W - target_w) // 2
    return img[:, top:top+target_h, left:left+target_w, :]


if __name__ == '__main__':
    lrhsi = sio.loadmat("./test_data_noref_small.mat")['hsi']

    ssr = sio.loadmat("/home/s-zhangbd/code/SSRnet_hmp/out/ssr_real_new_best.mat")['output']
    hsr = sio.loadmat("/home/s-zhangbd/code/HSRnet/out/hsr_real_new_psnr.mat")['output']
    scp = sio.loadmat("/home/s-zhangbd/code/SCPNet_hmp/out/scp_real_new.mat")['output']
    hyper = sio.loadmat("./hyper_real9_1_new.mat")['output']
    hmp = sio.loadmat("/home/s-zhangbd/code/HMPNet-master/fusion_tests/hmp_real_new_new_1.mat")['output']
    dsp = sio.loadmat("/home/s-zhangbd/code/DSP_hmp/out/dsp_real_best_new.mat")['output']
    psrt = sio.loadmat("/home/s-zhangbd/code/PSRT-main/result/hmp_real/psrt_real_best_new.mat")['output']
    dt3 = sio.loadmat("/home/s-zhangbd/code/3DT-Net_hmp/out/3dt_hmp_real_new.mat")['output']
    pmi = sio.loadmat("/home/s-zhangbd/code/PMI-RFCoNet-main/result_hmp/PMI_real_new.mat")['output']
    mimo = sio.loadmat("/home/s-zhangbd/code/MIMO-SST_hmp/out/MIMO_hmp_real__new_best.mat")['output']
    dpf = sio.loadmat("/home/s-zhangbd/code/DPFformer_hmp/Result_hmp/dpf_real_noref.mat")['output']  # DPF
    dim = sio.loadmat("/home/s-zhangbd/code/DIM-HMPF-main/DIM-HMPF/Result_hmp/real_2.mat")['output']   #DIM-HMP
    our = sio.loadmat("/home/s-zhangbd/code/work2_ablation/out_real/ours_real_noref_psnr.mat")['output'] # Ours


    index = ['ssr','hsr','scp','hyper','hmp','dsp','psrt','mimo','dt3','pmi','dpf','dim','our'] #['ssr','hsr','hyper','hmp','dsp','psrt','mimo','dt3','pmi','our']
    index = ['lrhsi']
    # 生成伪彩色图像
    for i in index:
        print("----processing----",i)
        im_out =  globals()[i]
        print(np.max(im_out),np.min(im_out))
        # im_out   = center_crop(im_out,   672, 672)
        print(im_out.shape)
        im_out = np.clip(im_out,0,1)
        print(np.max(im_out), np.min(im_out))

        show(im_out,id=0,name=i)
    # 生成残差图像
    # for i in index:
    #     print("----processing----",i)
    #     im_out =  globals()[i]
    #     print(np.max(im_out),np.min(im_out))
    #     im_out = np.clip(im_out,0,1)
    #     print(np.max(im_out), np.min(im_out))

    #     err_map(gt=gt,data=im_out,id=3,name=i,band=30)

