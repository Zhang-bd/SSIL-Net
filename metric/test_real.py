import numpy as np
import cv2


########################################################
# Gaussian degradation (approximate MTF)
########################################################
def mtf_downsample(img, scale, sigma=1.7):
    """
    img : (H,W)
    """
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape
    return cv2.resize(
        img,
        (w // scale, h // scale),
        interpolation=cv2.INTER_CUBIC,
    )


########################################################
# Universal Image Quality Index (UIQI)
########################################################
def qindex(img1, img2, block_size=8):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    h, w = img1.shape

    if min(h, w) < block_size:
        block_size = min(h, w)

    window = np.ones((block_size, block_size), dtype=np.float64)
    window /= window.size

    mu1 = cv2.filter2D(img1, -1, window, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.filter2D(img2, -1, window, borderType=cv2.BORDER_REFLECT)

    sigma1 = cv2.filter2D(img1 ** 2, -1, window,
                          borderType=cv2.BORDER_REFLECT) - mu1 ** 2
    sigma2 = cv2.filter2D(img2 ** 2, -1, window,
                          borderType=cv2.BORDER_REFLECT) - mu2 ** 2

    sigma12 = cv2.filter2D(
        img1 * img2,
        -1,
        window,
        borderType=cv2.BORDER_REFLECT,
    ) - mu1 * mu2

    eps = 1e-12

    q = ((2 * mu1 * mu2 + eps) *
         (2 * sigma12 + eps)) / \
        ((mu1 ** 2 + mu2 ** 2 + eps) *
         (sigma1 + sigma2 + eps))

    return np.mean(q)


########################################################
# D_lambda
########################################################
def D_lambda(fused, lr_hsi, scale, p=1):
    """
    fused : (C,H,W)
    lr_hsi: (C,h,w)
    """

    C = fused.shape[0]

    fused_lr = np.stack([
        mtf_downsample(band, scale)
        for band in fused
    ])

    diff = []

    for i in range(C):

        for j in range(i + 1, C):

            q_f = qindex(fused_lr[i], fused_lr[j])

            q_l = qindex(lr_hsi[i], lr_hsi[j])

            diff.append(abs(q_f - q_l) ** p)

    diff = np.array(diff)

    return (diff.mean()) ** (1 / p)


########################################################
# D_s
########################################################
def D_s(fused, lr_hsi, pan, scale, q=1):
    """
    fused : (C,H,W)
    lr_hsi: (C,h,w)
    pan : (H,W)
    """

    pan_lr = mtf_downsample(pan, scale)

    diff = []

    for i in range(fused.shape[0]):

        q_hr = qindex(fused[i], pan)

        q_lr = qindex(lr_hsi[i], pan_lr)

        diff.append(abs(q_hr - q_lr) ** q)

    diff = np.array(diff)

    return (diff.mean()) ** (1 / q)


########################################################
# QNR
########################################################
def QNR(
        fused,
        lr_hsi,
        pan,
        scale,
        alpha=1,
        beta=1):

    dl = D_lambda(fused, lr_hsi, scale)

    ds = D_s(fused, lr_hsi, pan, scale)

    return (1 - dl) ** alpha * (1 - ds) ** beta


########################################################
# SCC
########################################################
# def SCC(fused, pan):
#     """
#     fused : (C,H,W)
#     pan : (H,W)
#     """

#     gx_pan = cv2.Sobel(pan, cv2.CV_64F, 1, 0, 3)
#     gy_pan = cv2.Sobel(pan, cv2.CV_64F, 0, 1, 3)

#     pan_grad = np.sqrt(gx_pan ** 2 + gy_pan ** 2)

#     cc = []

#     for band in fused:

#         gx = cv2.Sobel(band, cv2.CV_64F, 1, 0, 3)
#         gy = cv2.Sobel(band, cv2.CV_64F, 0, 1, 3)

#         grad = np.sqrt(gx ** 2 + gy ** 2)

#         cc.append(np.corrcoef(
#             grad.ravel(),
#             pan_grad.ravel()
#         )[0, 1])

#     return np.mean(cc)

def SCC(fused, pan):
    """
    Spatial Correlation Coefficient (SCC)

    Parameters
    ----------
    fused : ndarray
        (C,H,W)
    pan : ndarray
        (H,W) or (1,H,W)

    Returns
    -------
    float
    """

    if pan.ndim == 3:
        pan = pan.squeeze(0)

    fused = fused.astype(np.float64)
    pan = pan.astype(np.float64)

    # ----------------------------
    # HSI -> pseudo PAN
    # ----------------------------
    fused_pan = np.mean(fused, axis=0)

    # ----------------------------
    # Sobel gradient
    # ----------------------------
    gx1 = cv2.Sobel(fused_pan, cv2.CV_64F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(fused_pan, cv2.CV_64F, 0, 1, ksize=3)
    grad1 = np.sqrt(gx1 ** 2 + gy1 ** 2)

    gx2 = cv2.Sobel(pan, cv2.CV_64F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(pan, cv2.CV_64F, 0, 1, ksize=3)
    grad2 = np.sqrt(gx2 ** 2 + gy2 ** 2)

    std1 = grad1.std()
    std2 = grad2.std()

    if std1 < 1e-12 or std2 < 1e-12:
        return 0.0

    return float(np.corrcoef(
        grad1.ravel(),
        grad2.ravel()
    )[0, 1])

def to_numpy(x):
    x = x.detach().cpu().numpy()

    while x.ndim > 3:
        x = np.squeeze(x, axis=0)

    return x
def evaluate_batch(fused, lr_hsi, pan):
    """
    fused  : (B,C,H,W)
    lr_hsi : (B,C,h,w)
    pan    : (B,1,H,W)

    Returns
    -------
    dict
    """

 

    B = fused.shape[0]

    scale = fused.shape[2] // lr_hsi.shape[2]

    metrics = {
        "D_lambda": [],
        "D_s": [],
        "QNR": [],
        "SCC": []
    }

    for i in range(B):

        f = fused[i]

        l = lr_hsi[i]

        p = pan[i, 0]


    

        metrics["D_lambda"].append(
            D_lambda(f, l, scale)
        )

        metrics["D_s"].append(
            D_s(f, l, p, scale)
        )

        metrics["QNR"].append(
            QNR(f, l, p, scale)
        )

        metrics["SCC"].append(
            SCC(f, p)
        )

    for k in metrics:

        metrics[k] = float(np.mean(metrics[k]))

    return metrics
def center_crop(img, target_h, target_w):
    """
    img: (N, H, W, C)
    """
    H, W = img.shape[1], img.shape[2]
    top = (H - target_h) // 2
    left = (W - target_w) // 2
    return img[:, top:top+target_h, left:left+target_w, :]
if __name__ == '__main__':
    import numpy as np
    import scipy.io as sio
    ###################################################
    # Example
    ###################################################
    # fused = sio.loadmat("/home/s-zhangbd/code/SSRnet_hmp/out/ssr_real_new_best.mat")['output']
    # fused = sio.loadmat("/home/s-zhangbd/code/HSRnet/out/hsr_real_new_psnr.mat")['output']
    # fused = sio.loadmat("/home/s-zhangbd/code/SCPNet_hmp/out/scp_real_new.mat")['output']
    # fused = sio.loadmat("./hyper_real9_1_new.mat")['output']
    # fused = sio.loadmat("/home/s-zhangbd/code/HMPNet-master/fusion_tests/hmp_real_new_new_1.mat")['output']
    # fused = sio.loadmat("/home/s-zhangbd/code/DSP_hmp/out/dsp_real_best_new.mat")['output']
    # fused = sio.loadmat("/home/s-zhangbd/code/PSRT-main/result/hmp_real/psrt_real_best_new.mat")['output']
    # fused = sio.loadmat("/home/s-zhangbd/code/3DT-Net_hmp/out/3dt_hmp_real_new.mat")['output']
    # fused = sio.loadmat("/home/s-zhangbd/code/PMI-RFCoNet-main/result_hmp/PMI_real_new.mat")['output']
    # fused = sio.loadmat("/home/s-zhangbd/code/MIMO-SST_hmp/out/MIMO_hmp_real__new_best.mat")['output']
    # fused = sio.loadmat("/home/s-zhangbd/code/DPFformer_hmp/Result_hmp/dpf_real_noref.mat")['output']  # DPF
    # fused = sio.loadmat("/home/s-zhangbd/code/DIM-HMPF-main/DIM-HMPF/Result_hmp/real_2.mat")['output']   #DIM-HMP
    fused = sio.loadmat("/home/s-zhangbd/code/work2_ablation/out_real/ours_real_noref_psnr.mat")['output'] # Ours


    lr_hsi = sio.loadmat("./test_data_noref_small.mat")['hsi']
    pan = sio.loadmat("./test_data_noref_small.mat")['pan']




    fused = fused.astype(np.float64)
    fused /= fused.max()

    lr_hsi = lr_hsi.astype(np.float64)
    lr_hsi /= lr_hsi.max()

    pan = pan.astype(np.float64)
    pan /= pan.max()

    fused = np.transpose(fused, (0, 3, 1, 2))
    lr_hsi = np.transpose(lr_hsi, (0, 3, 1, 2))
    pan = np.transpose(pan, (0, 3, 1, 2))

    print(fused.shape,lr_hsi.shape,pan.shape)

    metrics = evaluate_batch(
        fused,
        lr_hsi,
        pan
    )

    for k, v in metrics.items():
        print(f"{k:10s}: {v:.6f}")


