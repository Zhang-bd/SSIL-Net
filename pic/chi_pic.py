import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd


def show(data,id,name):
    t = np.squeeze(data[id])
    # band = (119-1, 90-1, 69-1)
    # band = (60-1,100-1,35-1)
    # band = 80
    band = (58 - 1, 100 - 1, 35 - 1)
    rgb = t[:, :, band]

    plt.imshow(rgb, cmap='gray')
    # plt.title('GT3')
    plt.axis('off')
    save_path = r'H:\work2\new_hsi_msi_pan\pic_false_color\CHI\pic2\0\\{}.png'.format(name)

    plt.imsave(save_path, rgb, cmap='gray')
    # plt.show()

def err_map(gt,data,id,name,band):
    output = np.squeeze(data[id])
    img_refference = np.squeeze(gt[id])
    plt.imshow(abs(output[:, :, band] - img_refference[:, :, band]), vmin=0, vmax=0.06, cmap='jet') #0.05
    plt.axis('off')
    save_path = r'H:\work2\new_hsi_msi_pan\pic_false_color\CHI\error2\0\\{}.png'.format(name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()
    return

def test_mri2(im_gt,im_out,name):
    sum_rmse, sum_sam, sum_psnr, sum_ssim, sum_ergas = [], [], [], [], []
    for i in range(im_gt.shape[0]):
        print(im_out[i].shape)
        score = quality_assessment(x_pred=im_out[i], x_true=im_gt[i], data_range=1, ratio=8, multi_dimension=False,
                                   block_size=8)
        sum_rmse.append(score['RMSE'])
        sum_psnr.append(score['MPSNR'])
        sum_ssim.append(score['MSSIM'])
        sum_sam.append(score['SAM'])
        sum_ergas.append(score['ERGAS'])
        print("----test{}------".format(i + 1))
        print("--PSNR", score['MPSNR'])
        print("--SAM", score['SAM'])
        print("--RMSE", score['RMSE'])
        print("--ERGAS", score['ERGAS'])
        print("--SSIM", score['MSSIM'])
    print()
    print("=====test_AVG====")
    print("PSNR: {:.4f}".format(np.mean(sum_psnr)))
    print("SAM: {:.4f}".format(np.mean(sum_sam)))
    print("RMSE: {:.4f}".format(np.mean(sum_rmse)))
    print("ERGAS: {:.4f}".format(np.mean(sum_ergas)))
    print("SSIM: {:.4f}".format(np.mean(sum_ssim)))

results_df = pd.DataFrame()
def test_mri(im_gt, im_out, name):
    global results_df
    sum_rmse, sum_sam, sum_psnr, sum_ssim, sum_ergas = [], [], [], [], []
    test_results = []  # Store results for each test

    for i in range(im_gt.shape[0]):
        print(im_out[i].shape)
        score = quality_assessment(x_pred=im_out[i], x_true=im_gt[i], data_range=1, ratio=8, multi_dimension=False,
                                   block_size=8)
        sum_rmse.append(score['RMSE'])
        sum_psnr.append(score['MPSNR'])
        sum_ssim.append(score['MSSIM'])
        sum_sam.append(score['SAM'])
        sum_ergas.append(score['ERGAS'])

        # Store individual test results
        test_results.append([score['MPSNR'], score['SAM'], score['RMSE'], score['ERGAS'], score['MSSIM']])

        print("----test{}------".format(i + 1))
        print("--PSNR", score['MPSNR'])
        print("--SAM", score['SAM'])
        print("--RMSE", score['RMSE'])
        print("--ERGAS", score['ERGAS'])
        print("--SSIM", score['MSSIM'])

    # Calculate and store average results
    avg_results = [
        np.mean(sum_psnr), np.mean(sum_sam), np.mean(sum_rmse), np.mean(sum_ergas), np.mean(sum_ssim)
    ]
    test_results.append(avg_results)

    print("\n=====test_AVG====")
    print("PSNR: {:.4f}".format(np.mean(sum_psnr)))
    print("SAM: {:.4f}".format(np.mean(sum_sam)))
    print("RMSE: {:.4f}".format(np.mean(sum_rmse)))
    print("ERGAS: {:.4f}".format(np.mean(sum_ergas)))
    print("SSIM: {:.4f}".format(np.mean(sum_ssim)))

    # Convert the results of all tests into a DataFrame
    test_df = pd.DataFrame(test_results, columns=['PSNR', 'SAM', 'RMSE', 'ERGAS', 'SSIM'])

    # Add the results of the current method (name) as a new column to the global DataFrame
    results_df[name] = test_df['PSNR']  # Add PSNR column for the current method
    results_df[f'{name}_SAM'] = test_df['SAM']  # Add SAM column
    results_df[f'{name}_RMSE'] = test_df['RMSE']  # Add RMSE column
    results_df[f'{name}_ERGAS'] = test_df['ERGAS']  # Add ERGAS column
    results_df[f'{name}_SSIM'] = test_df['SSIM']  # Add SSIM column
def save_results_to_excel(filename):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False, sheet_name="Comparison_Results")
if __name__ == '__main__':

    gt = sio.loadmat(r"H:\work2\测试数据\test.mat")['test']  # Chikusei

    # ssr = sio.loadmat(r"H:\work2\结果\SSRnet\ssr_chi_10000.mat")['output']  # ssrnet
    # hsr = sio.loadmat(r"H:\work2\结果\Hsr\hsr_chi_14550.mat")['output']  # hsrnet
    # scp = sio.loadmat(r"H:\work2\结果\SCPNet\out\scip_chi.mat")['output'] # scpnet
    # hyper = sio.loadmat(r"H:\work2\结果\Hyper\hyper2\hyper_Chi_256.mat")['output']  # hypernet new
    # hmp = sio.loadmat(r"H:\work2\结果\HMPNet\最新\out1_chi_256.mat")['out']  # HMP
    # dsp = sio.loadmat(r"H:\work2\结果\DSP\dsp_hmp_chi.mat")['output']  # dspnet
    # psrt = sio.loadmat(r"H:\work2\结果\PSRT\psrt_chi_1.mat")['output']  # psrt
    # dt3 = sio.loadmat(r"H:\work2\结果\3DT\3dt_hmp_chi_.mat")['output']  # 3dt
    # mimo = sio.loadmat(r"H:\work2\结果\MIMO\MIMO_hmp_chi_1.mat")['output']  #mimo-sst
    # pmi = sio.loadmat(r"H:\work2\结果\PMII\result_hmp\PMI_chi_1.mat")['output']  # PMI real
    dpf =  sio.loadmat(r"H:\work2\结果\DPFormer\chi_1.mat")['output']
    dim = sio.loadmat(r"H:\work2\结果\DIM\chi_2.mat")['output']
    # our = sio.loadmat(r"H:\work2\结果\Ours\new_model9_10_830.mat")['output']
    # our = sio.loadmat(r"H:\work2\结果\消融\m2_psnr.mat")['output'] # new m2
    # index = ['ssr','hsr', 'scp','hyper','hmp','dsp','psrt','mimo','dt3','pmi','our','gt'] #['ssr','hsr','hyper','hmp','dsp','psrt','mimo','dt3','pmi','our']
    index = ['dpf','dim']
    # 测试指标
    # from metirc import quality_assessment
    # for i in index:
    #     print("----processing----",i)
    #     im_out =  globals()[i]
    #     print(np.max(im_out),np.min(im_out))
    #     im_out = np.clip(im_out,0,1)
    #     print(np.max(im_out), np.min(im_out))
    #
    #     test_mri(gt,im_out,i)
    #
    # save_results_to_excel("comparison_results.xlsx")

    # 生成伪彩色图像
    for i in index:
        print("----processing----",i)
        im_out =  globals()[i]
        print(np.max(im_out),np.min(im_out))
        im_out = np.clip(im_out,0,1)
        print(np.max(im_out), np.min(im_out))

        show(im_out,id=0,name=i)


    # # 生成残差图像
    for i in index:
        print("----processing----",i)
        im_out =  globals()[i]
        print(np.max(im_out),np.min(im_out))
        im_out = np.clip(im_out,0,1)
        print(np.max(im_out), np.min(im_out))

        err_map(gt=gt,data=im_out,id=0,name=i,band=100) # 80 100

