# -*- coding: utf-8 -*-

"""
logo识别接口
"""
import cv2

from reg_main import reg_img_obj, ModelReg

__author__ = 'ZhangJiabao'

from flask import Blueprint, request, make_response, jsonify
from PIL import Image
from io import BytesIO

num = Blueprint('num', __name__)

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
import tensorflow as tf

e5 = cv2.getTickCount()
global modelReg
modelReg = ModelReg()
global graph
graph = tf.get_default_graph()
e6 = cv2.getTickCount()
the_time = (e6 - e5) / cv2.getTickFrequency()
print("===========模型初始化耗时： %s" % the_time)


@num.route('/detect', methods=['POST'])
def detect():
    e1 = cv2.getTickCount()
    file = request.get_data()
    if file is None:
        return make_response(jsonify({'error': '未能读到图片！'}), 400)
    try:
        image = Image.open(BytesIO(file))
    except IOError:
        return make_response(jsonify({'error': '读取图片出错，请确保传递了正确的图片！'}), 400)
    # image.show()

    # 先判断图片是否有exif信息
    # if hasattr(image, '_getexif'):
    #     # 获取exif信息
    #     dict_exif = image._getexif()
    #     if dict_exif(274, 0) == 3:
    #         # 旋转
    #         new_img = image.rotate(-90)
    #     elif dict_exif(274, 0) == 6:
    #         # 旋转
    #         new_img = image.rotate(180)
    #     else:
    #         new_img = image
    # else:
    #     new_img = image

    with graph.as_default():
        try:
            arr = reg_img_obj(image, modelReg)
        except Exception as e:
            errmsg = "出现错误:%s" % e
            print("=========================",errmsg)
            return make_response(jsonify({'error': errmsg}), 500)
    e2 = cv2.getTickCount()
    the_time = (e2 - e1) / cv2.getTickFrequency()
    print("===========识别耗时： %ss" % the_time)
    return jsonify(arr)


if __name__ == '__main__':
    pass
