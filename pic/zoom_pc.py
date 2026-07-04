import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
# 假设 `image_path` 是您的图像文件路径, 不能包含中文
# image_path = r'./CHI/error/0/our.png'
# path2 = r'./tmp/Ours_error.png'

def save_img(image_path,path2):
    # 使用cv2读取图像
    image = cv2.imread(image_path)
    # image_rgb = image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 获取图像的尺寸
    height, width, _ = image_rgb.shape

    # 定义要放大的区域（长方形）#(x, y, width, height)  横轴x, 竖轴y 左上角的点
    # red_box = (369* 160 //256, 369* 130//256, 369* 20 //256, 369* 20 //256) # 红框坐标，进一步向右下方移动
    # red_box = (160,130,20,20)
    # green_box = (100, 70, 100, 80)

    # red_box = (369 * 127 // 256, 369 * 80 // 256, 369 * 25 // 256, 369 * 25 // 256)  # 红框坐标，进一步向右下方移动
    # red_box = (369 * 140 // 256, 369 * 60 // 256, 369 * 25 // 256, 369 * 25 // 256)  # 红框坐标，进一步向右下方移动
    # red_box = (127,80,25,25)   # fasle
    # red_box = (369 * 170 // 256, 369 * 100 // 256, 369 * 40 // 256, 369 * 40 // 256)
    # red_box = (170, 100, 40, 40)  #error
    red_box = (118, 69, 28, 28)  #fasle
    # 从图像中提取这些区域
    red_region = image_rgb[red_box[1]:red_box[1]+red_box[3], red_box[0]:red_box[0]+red_box[2]]
    # green_region = image_rgb[green_box[1]:green_box[1]+green_box[3], green_box[0]:green_box[0]+green_box[2]]

    # 调整提取区域的大小为源图像的一半
    # new_size = (width // 2, height // 2)
    h, w, _ = red_region.shape
    new_size = (w*4, h*4)
    red_region_resized = cv2.resize(red_region, new_size, interpolation=cv2.INTER_LINEAR)
    # green_region_resized = cv2.resize(green_region, new_size, interpolation=cv2.INTER_LINEAR)

    # 创建一个新的图像用于显示结果
    # result_image = np.zeros((height , width, 3), dtype=np.uint8)

    # 在新图像上放置原图和放大的区域
    # result_image[:, :] = image_rgb
    # result_image[height:height + height // 2, :width // 2] = green_region_resized # 交换红框和绿框的位置
    # 获取red_region_resized的形状
    red_h, red_w, _ = red_region_resized.shape
    print(red_h, red_w, new_size)
    # 将red_region_resized复制到result_image的左下角区域
    image_rgb[-red_h:, :red_w] = red_region_resized

    # 在原图上使用矩形框标记放大区域
    cv2.rectangle(image_rgb, (red_box[0], red_box[1]), (red_box[0] + red_box[2], red_box[1] + red_box[3]), (255, 0, 0), 1)
    # cv2.rectangle(result_image, (green_box[0], green_box[1]), (green_box[0] + green_box[2], green_box[1] + green_box[3]), (0, 255, 0), 2)

    # 在放大的区域上也添加相应的框
    cv2.rectangle(image_rgb[-red_h:, :red_w], (0, 0), (red_w, red_h), (255, 0, 0), 2)
    # cv2.rectangle(image_rgb[-red_h:, :red_w], (0, 0), (red_h, red_w), (255, 0, 0), 1) # 交换红框和绿框的颜色
    # cv2.rectangle(result_image[height:height + height // 2, width // 2:width], (0, 0), (new_size[0], new_size[1]), (255, 0, 0), 2)

    # 保存结果图片为PNG格式
    # 将RGB图像转换回BGR格式
    # result_image_bgr = result_image
    result_image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    #保存结果图片为PNG格式
    cv2.imwrite(path2, result_image_bgr)

    # # 显示结果
    # plt.figure(figsize=(10, 10))
    # # plt.imshow(image_rgb)
    # plt.axis('off')
    # plt.show()

if __name__ == '__main__':
    index = ['ssr', 'scp','hsr', 'hyper', 'hmp', 'dsp', 'psrt', 'mimo', 'dt3', 'pmi', 'our', 'gt','dpf','dim']
    # index = ['dpf','dim']
    import os
    for i in index:
        img = os.path.join(r"H:\work2\new_hsi_msi_pan\pic_false_color\PC\pic_2", str(i)+'.png')  #pic_2 error_2
        # print(img)
        path2 = os.path.join(r"H:\work2\new_hsi_msi_pan\pic_false_color\PC\pic_3z", str(i)+'.png')
        save_img(img,path2)