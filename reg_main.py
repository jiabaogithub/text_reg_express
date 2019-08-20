# coding:UTF-8

import cv2
import numpy as np
from PIL import Image

from crnn_reg.crnn_predict import init_crnn_model, pred
from loc import cfg
from loc.network import East
from loc.predict_batch import predict


class ModelReg():

    def __init__(self):
        # 文本定位模型初始化
        east = East()
        self.text_loc = east.east_network()
        self.text_loc.load_weights(cfg.saved_model_weights_file_path)
        # 数字识别模型初始化
        # self.reg_model_num = keras.models.load_model('reg/checkpoint/num_32size-20190703.h5')
        # crnn 模型初始化
        self.crnn_model_mobile = init_crnn_model("crnn_reg/trained_models/netCRNN_mobiles_26w_0.99acc.pth")
        # self.crnn_model_mobile = init_crnn_model("crnn_reg/trained_models/netCRNN_mobiles_0.94acc.pth")
        self.crnn_model_all = init_crnn_model("crnn_reg/trained_models/netCRNN_all_200w_0.829acc.pth")
        # self.crnn_model_all = init_crnn_model("crnn_reg/trained_models/netCRNN_all_80acc-best.pth")


# 定位图片目标区域
def loc(text_loc, img, threshold):
    rs_xy = predict(text_loc, img, threshold, save=True)
    return rs_xy


def check_divide(y_chars_dict):
    for lt_y, imgSegments in y_chars_dict.items():
        for i, imgSeg in enumerate(imgSegments):
            # cv2.imshow('imgSeg', imgSeg)
            # cv2.waitKey(0)
            pass


def crnn_reg(loced_imgs, crnn_model_mobile, crnn_model_all, save=False):
    mobiles_dict = {}
    for i, img_obj in enumerate(loced_imgs):
        img_y = img_obj[0]
        cut_img = img_obj[1]
        cut_img_arr = np.array(cut_img)
        # cut_img_arr = cv2.cvtColor(cut_img_arr, cv2.COLOR_RGB2BGR)
        cut_img_arr = cv2.cvtColor(cut_img_arr, cv2.COLOR_RGB2GRAY)
        # 二值化 21，10
        cut_img_arr = cv2.adaptiveThreshold(cut_img_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 25, 6)
        if save:
            cv2.imwrite("img_gen/%s_%s_cut.jpg" % (i, img_y), cut_img_arr)

        # 转RGB
        cut_img_arr = cv2.cvtColor(cut_img_arr, cv2.COLOR_GRAY2RGB)
        # 预测是否是手机号
        rs = pred(crnn_model_all, cut_img_arr)
        # if len(rs) == 11 and rs.startswith("1"):
        if len(rs) >=10 and len(rs) <=12 :
            # 预测手机号码
            rs = pred(crnn_model_mobile, cut_img_arr)
            mobiles_dict[img_y] = rs
        else:
            continue
    return mobiles_dict


# 滤波，去除噪点
def try_smoothing(img_obj):
    # img_obj.save("img_gen/" + '_before_blur.jpg')
    # 二值化
    img_gray_arr = cv2.cvtColor(np.array(img_obj), cv2.COLOR_RGB2GRAY)
    img_bin_arr = cv2.adaptiveThreshold(img_gray_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 25, 6)
    # cv2.imwrite("img_gen/" + '_before_blur_bin.jpg',img_bin_arr)
    img_bin_arr = cv2.medianBlur(img_bin_arr, 5)
    img_obj = Image.fromarray(img_bin_arr)
    # print("保存滤波后的图片")
    # img_obj.save("img_gen/" + '_blur.jpg')
    return img_obj


def reg_img_obj(img_obj, modelReg, threshold=0.8):
    e1 = cv2.getTickCount()
    text_loc = modelReg.text_loc
    # reg_model_num = modelReg.reg_model_num
    crnn_model_mobile = modelReg.crnn_model_mobile
    crnn_model_all = modelReg.crnn_model_all
    arr = {}
    # 定位图片目标区域
    regions_obj = loc(text_loc, img_obj, threshold)
    if len(regions_obj) == 0:
        print("检测目标未能成功，请重新给定样本！")
        # print("首次检测目标未能成功，正在尝试滤波后再次检测...")
        # # 滤波，去除噪点
        # img_obj = try_smoothing(img_obj)
        # # 定位图片目标区域
        # regions_obj = loc(text_loc, img_obj, threshold)
        # if len(regions_obj) == 0:
        #     print("二次处理后依然未能成功检测，请重新给定样本！")
    else:
        # crnn识别
        arr = crnn_reg(regions_obj, crnn_model_mobile, crnn_model_all)
        if len(arr) == 0:
            # print("识别手机号失败！")
            print("首次未能识别到手机号，正在尝试滤波后再次检测...")
            # 滤波，去除噪点
            img_obj = try_smoothing(img_obj)
            # 定位图片目标区域
            regions_obj = loc(text_loc, img_obj, threshold)
            # crnn识别
            arr = crnn_reg(regions_obj, crnn_model_mobile, crnn_model_all)
            if len(arr) == 0:
                print("二次识别手机号失败！")
        else:
            print(arr)
    e2 = cv2.getTickCount()
    the_time = (e2 - e1) / cv2.getTickFrequency()
    print("===========识别耗时： %s" % the_time)
    return arr
    # else:
    #     # 分割文字
    #     num = 11  # 指定文本目标的位数，比如手机号是11位
    #     width_ceiling = 25  # 指定手机号单个数字的宽度上限，面单场景中经过粗略统计应该为25
    #     y_chars_dict = divide_obj(regions_obj, num, width_ceiling)
    #
    #     # 检查分割情况
    #     # check_divide(y_chars_dict)
    #     if len(y_chars_dict) == 0:
    #         print("检测手机号失败！")
    #     else:
    #         # 识别收件人手机号
    #         for y, target in y_chars_dict.items():
    #             mobile_nums = reg_obj(target, reg_model_num, save=False)
    #             # 打印结果
    #             if len(mobile_nums) != 0:
    #                 print(y, mobile_nums)
    #                 arr[y] = "".join(mobile_nums)
    #         # 分开收件人和寄件人的手机目标：lt_y小的是收件人
    #         # y_num_dict_sorted = dict(sorted(arr.items(), key=lambda x: x[0], reverse=False))
    #         # print("=================== 收件人手机号：%s ====================", list(y_num_dict_sorted.values())[0])
    #         e2 = cv2.getTickCount()
    #         the_time = (e2 - e1) / cv2.getTickFrequency()
    #         print("===========识别耗时： %s" % the_time)
    # return arr


if __name__ == '__main__':
    import os

    if os.name == 'nt':
        pass
    else:
        # pass
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    # 模型初始化
    _e1 = cv2.getTickCount()
    modelReg = ModelReg()
    _e2 = cv2.getTickCount()
    the_time = (_e2 - _e1) / cv2.getTickFrequency()
    print("===========模型初始化耗时： %s" % the_time)

    # 图片路径
    # img_dir = 'img_train'
    # img_dir = 'img_test'
    img_dir = 'img'
    imgsList = os.listdir(img_dir)
    for img_ in imgsList:
        _img_full_name = img_
        print("图片名称： ", _img_full_name)
        img_path = os.path.join(img_dir, _img_full_name)
        img_obj = Image.open(img_path)
        reg_img_obj(img_obj, modelReg)
