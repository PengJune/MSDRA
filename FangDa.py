import cv2
import numpy as np
from PIL import Image

def get_mask_image(mask, left_top, right_top, left_bottom, right_bottom):
    # 显示anchor的图像 顺序必须为左上，左下，右下，右上
    contours = np.array([[left_top, left_bottom, right_bottom, right_top]], dtype=np.int)
    # print(contours)
    """
    第一个参数是显示在哪个图像上；
    第二个参数是轮廓；
    第三个参数是指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓；
    第四个参数是绘制的颜色；
    第五个参数是线条的粗细
    """
    mask_image = cv2.drawContours(mask, contours, -1, (0, 0, 255), 2)  # 颜色：BGR
    # cv2.imshow('drawimg', mask_image)
    # cv2.waitKey(0)
    return mask_image

if __name__ == '__main__':
    image_path = "D:/fusion\pfuse\imagess\mri-spect-1-1\COT/SPECT1.jpg" # 加载某张图像

    original_image = cv2.imread(image_path)
    # print(original_image.shape)
    original_image_width = original_image.shape[1]
    original_image_height = original_image.shape[0]
    print("该图像尺寸(宽*高)为：{}*{}".format(original_image_width, original_image_height))

    left_top = [50,50] # anchor左上角的坐标
    right_top = [80,50] # anchor右上角的坐标
    left_bottom = [50,100] # anchor左下角的坐标
    right_bottom = [80,100] # anchor右下角的坐标

    mask = original_image.copy()
    mask_image = get_mask_image(mask, left_top, right_top, left_bottom, right_bottom)

    x1 = min(left_top[0], right_top[0], left_bottom[0], right_bottom[0])
    x2 = max(left_top[0], right_top[0], left_bottom[0], right_bottom[0])
    y1 = min(left_top[1], right_top[1], left_bottom[1], right_bottom[1])
    y2 = max(left_top[1], right_top[1], left_bottom[1], right_bottom[1])
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_image[y1:y1 + hight, x1:x1 + width] # 得到剪切后的图像
    # print(crop_img.shape)
    # cv2.imshow('cuttimg', crop_img)
    # cv2.waitKey(0)

    img = Image.fromarray(crop_img)
    # 这里如果没有mask直接操作原图，那么剪切后的图像会带个蓝框
    # 因为上边生成mask_image的时候颜色顺序是BGR，但是这里是RGB
    # img.show()

    img = img.resize((original_image_width, original_image_height))
    # img.show()

    # 给放大的图加红色框
    left_top = [0, 0]  # anchor左上角的坐标
    right_top = [original_image_width, 0]  # anchor右上角的坐标
    left_bottom = [0, original_image_height]  # anchor左下角的坐标
    right_bottom = [original_image_width, original_image_height]  # anchor右下角的坐标
    img = np.array(img)
    mask_crop_img = get_mask_image(img, left_top, right_top, left_bottom, right_bottom)

    result_img = np.vstack((mask_image, mask_crop_img))
    cv2.imwrite("D:/fusion\pfuse\imagess\mri-spect-1-1\COT/result1.jpg", result_img)

