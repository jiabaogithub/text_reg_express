# coding:UTF-8
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from contours_utils import fill_255_for_other
from loc import cfg
from loc.label import point_inside_of_quad
from loc.network import East
from loc.nms import nms
from loc.preprocess import resize_image


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)


def predict(east_detect, img_obj, pixel_threshold, quiet=False, save=False):
    img = img_obj
    # cv2.imshow('img', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    # imS = cv2.resize(np.array(img), (416, 416))
    # cv2.imshow('imS', cv2.cvtColor(imS,cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)

    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    # with Image.open(img_path) as im:
    # im_array = image.img_to_array(im.convert('RGB'))
    im = img_obj
    d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
    im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    # im = im.resize((d_wight, d_height), Image.ANTIALIAS).convert('RGB')
    org_quad_im = im.copy()
    quad_im = im.copy()
    draw = ImageDraw.Draw(im)
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        px = (j + 0.5) * cfg.pixel_size
        py = (i + 0.5) * cfg.pixel_size
        line_width, line_color = 1, 'red'
        if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
            if y[i, j, 2] < cfg.trunc_threshold:
                line_width, line_color = 2, 'yellow'
            elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                line_width, line_color = 2, 'green'
        draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                   (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                   (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                   (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                   (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                  width=line_width, fill=line_color)
    if save:
        im.save("img_gen/" + '_act.jpg')
    quad_draw = ImageDraw.Draw(quad_im)
    rs_xy = []
    for score, geo, s in zip(quad_scores, quad_after_nms,
                             range(len(quad_scores))):
        if np.amin(score) > 0:
            if cfg.predict_cut_text_line:
                quad_im_array = np.array(quad_im)
                # 以左上角和右下角的坐标来锁定剪切位置
                lt_y = int(tuple(geo[0])[1])
                lt_x = int(tuple(geo[0])[0])
                lb_y = int(tuple(geo[1])[1])
                lb_x = int(tuple(geo[1])[0])
                rb_y = int(tuple(geo[2])[1])
                rb_x = int(tuple(geo[2])[0])
                rt_y = int(tuple(geo[3])[1])
                rt_x = int(tuple(geo[3])[0])
            quad_draw.line([tuple(geo[0]),
                            tuple(geo[1]),
                            tuple(geo[2]),
                            tuple(geo[3]),
                            tuple(geo[0])], width=2, fill='red')
            if lt_y <= 0 or lt_x <= 0 or lb_y <= 0 or lb_x <= 0 or rb_y <= 0 or rb_x <= 0 or rt_y <= 0 or rt_x <= 0:
                continue
            # if rb_y - lt_y > 0 and rb_x - lt_x > 50 and lt_x > 50 and rb_x - lt_x < quad_im_array.shape[1] // 2:  # 只获取宽度小于图片宽度1/2的目标
            # if lt_x > 50 and rb_x - lt_x < quad_im_array.shape[1] // 2:  # 只获取宽度小于图片宽度1/2的目标
            if lt_x > 1 :  # 小框拍照时使用
                # print(lt_y, rb_y, lt_x, rb_x)
                add_num = 1
                up_boundary = min(lt_y, rt_y)  # 上边界
                bottom_boundary = max(lb_y, rb_y)  # 下边界
                left_boundary = min(lt_x, lb_x)  # 左边界
                right_boundary = max(rb_x, rt_x)  # 右边界
                # 获取矩形坐标轮廓
                var_num = 2
                _contours = [np.array([[lt_x-var_num, lt_y-var_num], [lb_x-var_num, lb_y+var_num], [rb_x+var_num, rb_y+var_num], [rt_x+var_num, rt_y-var_num]])]
                # _contours = [np.array([[lt_x, lt_y], [lb_x, lb_y], [rb_x, rb_y], [rt_x, rt_y]])]
                # _contours = [np.array([[lt_y, lt_x], [lb_y, lb_x], [rb_y, rb_x], [rt_y, rt_x]])]
                # if rb_x + add_num <= quad_im_array.shape[1] and rt_x + add_num <= quad_im_array.shape[
                #     1]:  # 确保放宽后不会超过图片宽度
                #     left_boundary = min(lt_x, lb_x) - add_num  # 左边界，放宽add_num个像素，保证截取完整
                #     right_boundary = max(rb_x + add_num, rt_x + add_num)  # 右边界，放宽add_num个像素，保证截取完整
                #     # 获取矩形坐标轮廓
                #     _contours = [np.array([[lt_x - add_num, lt_y - add_num], [lb_x - add_num, lb_y + add_num],
                #                            [rb_x + add_num, rb_y + add_num], [rt_x + add_num, rt_y - add_num]])]

                # 目标轮廓之外填充黑色图像，以减少噪音
                org_quad_im_arr = cv2.cvtColor(np.array(org_quad_im),cv2.COLOR_RGB2BGR)
                cut_img_arr = fill_255_for_other(org_quad_im_arr, _contours,show=False)
                if cut_img_arr is None:
                    continue
                # cv2.imshow('cut_img_arr2', cut_img_arr)
                # cv2.waitKey(0)
                cut_img = Image.fromarray(cut_img_arr)
                rs_xy.append([lt_y, cut_img])
                if save:
                    # cv2.imwrite("img_gen/%s_%s_org_cut.jpg" % (lt_x, lt_y), cut_img_arr)
                    cut_img_arr = cv2.cvtColor(cut_img_arr, cv2.COLOR_RGB2GRAY)
                    # 二值化
                    cut_img_arr = cv2.adaptiveThreshold(cut_img_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, 25, 6)
                    cv2.imwrite("img_gen/%s_%s_cut.jpg" % (lt_x, lt_y), cut_img_arr)
                    # cut_img.save("img_gen/%s_%s_cut.jpg" % (lt_x, lt_y))
        elif not quiet:
            pass
            # print('quad invalid with vertex num less then 4.')
    if save:
        quad_im.save("img_gen/" + '_predict.jpg')
        img_obj.save("img_gen/" + '_org.jpg')
    return rs_xy


def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    if cfg.predict_write2txt and len(txt_items) > 0:
        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='demo/012.png',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


if __name__ == '__main__':
    import os

    if os.name == 'nt':
        pass
    else:
        # pass
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print(img_path, threshold)

    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)
    predict(east_detect, img_path, threshold)
