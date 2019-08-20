# coding:UTF-8
import time

import cv2
import numpy as np

from reg.alphabets import alphabet_num


def getBinary(oriImgPath):
    print("--------", oriImgPath)
    oriImg = cv2.imread(oriImgPath)
    # oriImg = cv2.resize(oriImg, (32, 32))
    oriImgGray = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)
    ret, thr = cv2.threshold(oriImgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr


# 查找字体的最小包含矩形
class FindImageBBox(object):
    def __init__(self, ):
        pass

    def do(self, img):
        height = img.shape[0]
        width = img.shape[1]
        v_sum = np.sum(img, axis=0)
        h_sum = np.sum(img, axis=1)
        left = 0
        right = width - 1
        top = 0
        low = height - 1
        # 从左往右扫描，遇到非零像素点就以此为字体的左边界
        for i in range(width):
            if v_sum[i] > 0:
                left = i
                break
        # 从右往左扫描，遇到非零像素点就以此为字体的右边界
        for i in range(width - 1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break
        # 从上往下扫描，遇到非零像素点就以此为字体的上边界
        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break
        # 从下往上扫描，遇到非零像素点就以此为字体的下边界
        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                low = i
                break
        return (left, top, right, low)


def reg_obj(chars_img_arr, reg_model_num,save=False):
    char_arr = []
    for i, char_img_arr in enumerate(chars_img_arr):
        binaryImg = char_img_arr  # char_img_arr已经是二值化图片
        # cv2.imshow('win2', binaryImg)
        # cv2.waitKey(0)
        # 切割非字符区域
        find_image_bbox = FindImageBBox()
        cropped_box = find_image_bbox.do(binaryImg)
        left, upper, right, lower = cropped_box
        binaryImg = binaryImg[upper: lower + 1, left: right + 1]
        # cv2.imshow('win2', binaryImg)
        # cv2.waitKey(0)

        model = reg_model_num  # 默认采用数字模型
        alphabet = alphabet_num
        sw = 32
        sh = 32

        # 调整大小
        binaryImg = cv2.resize(binaryImg, (sw, sh))
        # cv2.imshow('win4', binaryImg)
        # cv2.waitKey(0)

        # 调整维度
        _img = binaryImg.reshape((1, binaryImg.shape[0], binaryImg.shape[1], 1))
        # 预测
        yhat = model.predict(_img)[0]
        # print(yhat)
        # print(type(yhat))
        id = np.where(yhat == max(yhat))[0][0]
        # print("id:%s; val:%s" % (id, alphabet[id]))
        rs_num = alphabet[id]
        # 保存样本
        # if i ==0 and rs_num != "1": # 首数字未识别成1的就过滤掉
        #     print("------------首数字：%s",rs_num)
        #     break
        if save:
            millis = int(round(time.time() * 1000))
            import os
            path = os.path.join("train", rs_num)
            # path = os.path.join("test", rs_num)
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite("%s/%s_%s.jpg" % (path, millis, i), char_img_arr)
        char_arr.append(rs_num)
    return char_arr


if __name__ == '__main__':
    pass
