import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.ticker as ticker
# 计算均值光谱差异
def compute_mean_spectral_difference(I_recon, I_GT):
    spectral_diff = np.abs(I_recon - I_GT)  # 逐像素绝对差异
    mean_spectral_diff = np.mean(spectral_diff, axis=(0, 1))  # 对 H, W 维度求均值
    return mean_spectral_diff

# 加载数据

gt = sio.loadmat(r"../Real\data2\test_data.mat")['gt']  # Cave  (12,512,512,31)
methods = {
    "SSRNet": sio.loadmat(r"H:\work2\结果\result_real\ssr_real_best.mat")['output'],  # ssrnet
    "HSRNet": sio.loadmat(r"H:\work2\结果\result_real\hsr_real_psnr.mat")['output'],  # hsrnet
    "SCPNet": sio.loadmat(r"H:\work2\结果\SCPNet\out\scp_real.mat")['output'], # scpnet
    "HyperNet": sio.loadmat(r"H:\work2\结果\Hyper\hyper2\hyper_real9_1.mat")['output'],  # hyper new
    "HMPNet": sio.loadmat(r"H:\work2\结果\result_real\hmp_real_new_1.mat")['output'] , # hmpnet
    "DSPNet": sio.loadmat(r"H:\work2\结果\result_real\dsp_real_best.mat")['output'],  # dspnet
    "PSRT": sio.loadmat(r"H:\work2\结果\result_real\psrt_real_best.mat")['output'] , # psrt
    "3DT-Net": sio.loadmat(r"H:\work2\结果\result_real\3dt_hmp_real_91.mat")['output'] , # Cave 3dt
    "PMI-RFCoNet": sio.loadmat(r"H:\work2\结果\PMII\result_hmp\PMI_real_3_2_50.mat")['output'] , # PMI real
    "MIMO-SST": sio.loadmat(r"H:\work2\结果\result_real\MIMO_hmp_real_best.mat")['output'] , # mimo-sst
    # "Ours": sio.loadmat(r"H:\work2\结果\result_real\ours_model910_real_best.mat")['output']  # Cave ours
    "Ours": sio.loadmat(r"H:\work2\结果\Ours_dc\m2\m2_real_psnr_.mat")['output']  # Cave ours

}


for index in range(gt.shape[0]):
    # 绘制曲线
    plt.figure(figsize=(7, 5))
    # colors = plt.cm.get_cmap("tab10", len(methods))  # 生成不同颜色
    # print(colors)
    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_color("gray")
    ax.xaxis.set_tick_params(color="gray")  # 仅修改 x 轴刻度线颜色
    ax.yaxis.set_tick_params(color="gray")  # 仅修改 y 轴刻度线颜色
    ax.tick_params(
        axis="both",
        direction="in",  # 让刻度向内
        top=True, bottom=True, left=True, right=True  # 启用四周刻度
    )
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))  # 强制使用 10⁻³
    ax.yaxis.set_major_formatter(formatter)
    colors = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
        '#DAA520',
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
    ]
    # colors = ['','','','','','','','','','']
    marker = ['+','x','.','p','h','d','^','s','v','*','o']
    for i, (name, recon) in enumerate(methods.items()):
        mean_diff = compute_mean_spectral_difference(recon[index], gt[index])
        plt.plot(mean_diff, marker=marker[i], linestyle='-', color=colors[i], label=name,markersize=3,markevery=3)#markerfacecolor="white"
    # plt.yticks(np.arange(0, 0.07, step=0.01))
    plt.tick_params(direction="in")
    plt.xlim([0, 75])
    plt.xticks(np.arange(0, 76, step=8), np.arange(1, 77, step=8))  # 调整刻度标签
    plt.xlabel("Band number")
    plt.ylabel("Mean difference")
    # plt.title("Mean Spectral Difference Curve")
    plt.legend(loc="upper left",frameon=False,fontsize=9, borderpad=0.9, handlelength=2)
    plt.grid(False)
    # plt.show()
    plt.savefig("./real/real_{}_m2.pdf".format(index), format="pdf", dpi=600, bbox_inches="tight",pad_inches=1)
