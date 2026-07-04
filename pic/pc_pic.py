import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def show(data,id,name):
    t = np.squeeze(data[id])
    # band = (64-1, 32-1, 11-1)
    band = (66 - 1, 32 - 1, 11 - 1)
    # band = 80
    rgb = t[:, :, band]

    plt.imshow(rgb, cmap='gray')
    # plt.title('GT3')
    plt.axis('off')
    save_path = r'H:\work2\new_hsi_msi_pan\pic_false_color\PC\pic_2\\{}.png'.format(name)

    plt.imsave(save_path, rgb, cmap='gray')
    # plt.show()

def err_map(gt,data,id,name,band):
    output = np.squeeze(data[id])
    img_refference = np.squeeze(gt[id])
    plt.imshow(abs(output[:, :, band] - img_refference[:, :, band]), vmin=0, vmax=0.14, cmap='jet')
    plt.axis('off')
    save_path = r'H:\work2\new_hsi_msi_pan\pic_false_color\PC\error_2\\{}.png'.format(name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()
    return



if __name__ == '__main__':

    gt = sio.loadmat(r"../PC\data4\test_gt.mat")['gt']  # PC  (3, 256, 256, 93)
    #
    # ssr = sio.loadmat(r"H:\work2\结果\result_pc\ssr_pc_best.mat")['output']  # PC ssrnet
    # hsr = sio.loadmat(r"H:\work2\结果\result_pc\hsr_pc_psnr.mat")['output']  # PC hsrnet
    # # im_out = sio.loadmat(r"H:\work2\结果\result_pc\hyper_PC_551.mat")['output']  # PC hypernet
    # scp = sio.loadmat(r"H:\work2\结果\SCPNet\out\scp_pc.mat")['output']
    # hyper = sio.loadmat(r"H:\work2\结果\Hyper\hyper2\hyper_PC_551.mat")['output']  # PC hypernet
    #
    # hmp = sio.loadmat(r"H:\work2\结果\result_pc\hmp_pc_new_1.mat")['output']  # PC hmpnet
    # dsp = sio.loadmat(r"H:\work2\结果\result_pc\dsp_pc_best.mat")['output']  # PC dspnet
    # psrt = sio.loadmat(r"H:\work2\结果\result_pc\psrt_pc_1520.mat")['output']  # PC psrt
    # mimo = sio.loadmat(r"H:\work2\结果\result_pc\MIMO_hmp_pc_best.mat")['output']  # PC mimo-sst
    # dt3 = sio.loadmat(r"H:\work2\结果\result_pc\3dt_hmp_pc_.mat")['output']  # PC 3dt
    # our = sio.loadmat(r"H:\work2\结果\result_pc\pc_model8_13_best.mat")['output']  # PC ours
    # # our = sio.loadmat(r"H:\work2\结果\Ours_dc\m2\m2_pc_psnr.mat")['output']
    # pmi = sio.loadmat(r"H:\work2\结果\PMII\result_hmp\PMI_pc_1.mat")['output']  # PMI pc
    dpf = sio.loadmat(r"H:\work2\结果\DPFormer\pc_1.mat")['output']
    dim = sio.loadmat(r"H:\work2\结果\DIM\pc_2.mat")['output']
    # index = ['ssr','hsr','scp','hyper','hmp','dsp','psrt','mimo','dt3','pmi','our','gt','dpf','dim'] #['ssr','hsr','hyper','hmp','dsp','psrt','mimo','dt3','pmi','our']
    index = ['dpf','dim']
    # 生成伪彩色图像
    for i in index:
        print("----processing----",i)
        im_out =  globals()[i]
        print(np.max(im_out),np.min(im_out))
        im_out = np.clip(im_out,0,1)
        print(np.max(im_out), np.min(im_out))

        show(im_out,id=1,name=i)
    # # 生成残差图像
    for i in index:
        print("----processing----",i)
        im_out =  globals()[i]
        print(np.max(im_out),np.min(im_out))
        im_out = np.clip(im_out,0,1)
        print(np.max(im_out), np.min(im_out))

        err_map(gt=gt,data=im_out,id=1,name=i,band=70) #80

