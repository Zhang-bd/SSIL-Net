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

gt = sio.loadmat(r"../PC\data4\test_gt.mat")['gt']  # PC  (3, 256, 256, 93)
methods = {
    "SSRNet": sio.loadmat(r"H:\work2\结果\result_pc\ssr_pc_best.mat")['output'],  # PC ssrnet
    "HSRNet": sio.loadmat(r"H:\work2\结果\result_pc\hsr_pc_psnr.mat")['output'],  # PC hsrnet
    "SCPNet": sio.loadmat(r"H:\work2\结果\SCPNet\out\scp_pc.mat")['output'], # scpnet
    "HyperNet": sio.loadmat(r"H:\work2\结果\Hyper\hyper2\hyper_PC_551.mat")['output'],  # PC hypernet
    "HMPNet": sio.loadmat(r"H:\work2\结果\result_pc\hmp_pc_new_1.mat")['output'], # PC hmpnet
    "DSPNet": sio.loadmat(r"H:\work2\结果\result_pc\dsp_pc_best.mat")['output'],  # PC dspnet
    "PSRT": sio.loadmat(r"H:\work2\结果\result_pc\psrt_pc_1520.mat")['output'],  # PC psrt
    "3DT-Net": sio.loadmat(r"H:\work2\结果\result_pc\3dt_hmp_pc_.mat")['output'],  # PC 3dt
    "PMI-RFCoNet": sio.loadmat(r"H:\work2\结果\PMII\result_hmp\PMI_pc_1.mat")['output'],  # PMI pc
    "MIMO-SST": sio.loadmat(r"H:\work2\结果\result_pc\MIMO_hmp_pc_best.mat")['output'],  # PC mimo-sst
    "DPFormer" : sio.loadmat(r"H:\work2\结果\DPFormer\pc_1.mat")['output'],
    "DIM-HMPF": sio.loadmat(r"H:\work2\结果\DIM\pc_2.mat")['output'],
    "Ours": sio.loadmat(r"H:\work2\结果\result_pc\pc_model8_13_best.mat")['output']  # PC ours
    # "Ours": sio.loadmat(r"H:\work2\结果\Ours_dc\m2\m2_pc_psnr.mat")['output']  # PC ours  M2
}


for index in range(1,2):
    # 绘制曲线
    plt.figure(figsize=(7, 5))
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
    # colors = plt.cm.get_cmap("tab10", len(methods))  # 生成不同颜色
    # print(colors)
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
    '#C67171',
    '#7A378B',
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
    ]
    # colors = ['','','','','','','','','','']
    marker = ['+','x','.','p','h','d','^','s','v','*','o']
    for i, (name, recon) in enumerate(methods.items()):
        mean_diff = compute_mean_spectral_difference(recon[index], gt[index])
        plt.plot(mean_diff, linestyle='-', color=colors[i], label=name,markersize=4,markevery=3)#markerfacecolor="white"marker=marker[i]
    plt.tick_params(direction="in")
    # plt.tick_params(axis="both", colors="gray")  # 修改 x 和 y 轴刻度的颜色
    plt.yticks(np.arange(0.005, 0.020, step=0.002))
    plt.xlim([0, 92])
    plt.xticks(np.arange(0, 93, step=13), np.arange(1, 94, step=13))  # 调整刻度标签
    plt.xlabel("Band number")
    plt.ylabel("Mean difference")
    # plt.title("Mean Spectral Difference Curve")
    plt.legend(loc="upper left",frameon=False,fontsize=10, borderpad=0.9, handlelength=2,ncol=2)
    plt.grid(False)

    plt.savefig("./pc/pc_new_{}.pdf".format(index), format="pdf", dpi=600, bbox_inches="tight",pad_inches=1)
    plt.show()
