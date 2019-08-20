import cv2
import numpy as np

from reg.char_divide import horizontalDivide, verticalDivide
from reg.text_preprocess import fix_angle


def fill_255_for_other(img, contours, show=False):
    if show:
        cv2.imshow('img', img)
        cv2.waitKey(0)

    stencil = np.zeros(img.shape).astype(img.dtype)
    # stencil = stencil+255
    # color = [0, 0, 0]
    color = [255, 255, 255]
    cv2.fillPoly(stencil, contours, color)

    if show:
        cv2.imshow('stencil', stencil)
        cv2.waitKey(0)

    result = cv2.bitwise_and(img, stencil)
    # result = cv2.add(img, stencil)
    if show:
        cv2.imshow('result', result)
        cv2.waitKey(0)

    # 二值化
    imagegray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    retval, imagebin = cv2.threshold(imagegray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # imagebin = cv2.adaptiveThreshold(imagegray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #                                  15, 10)
    if show:
        cv2.imshow('imagebin', imagebin)
        cv2.waitKey(0)
    rotated = fix_angle(imagebin, result)
    # 二值化
    imagegray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
    retval, imagebin_rotated = cv2.threshold(imagegray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, t, b = horizontalDivide(imagebin_rotated)
    if len(t) == 0 or len(b) == 0:
        return None
    imagebin_rotated_seg = imagebin_rotated[t[0]:b[0], :]
    _, l, r = verticalDivide(imagebin_rotated_seg)
    if len(l) == 0 or len(r) == 0:
        return None
    rotated_seg = rotated[t[0] + 4:b[0] - 2, l[0] + 2:r[0] - 2]  # 向内缩小var_numm步
    # rotated_seg = rotated[t[0]+var_num:b[0]-var_num,l[0]:r[0]] # 向内缩小var_numm步
    if rotated_seg.shape[0] <= 0 or rotated_seg.shape[1] <= 0:
        return None

    if show:
        cv2.imshow('imagebin', imagebin)
        cv2.imshow('rotated', rotated)
        cv2.imshow('imagebin_rotated', imagebin_rotated)
        cv2.imshow('rotated_seg', rotated_seg)
        cv2.waitKey(0)

    return rotated_seg


if __name__ == '__main__':
    # _img = cv2.imread('cut/_org.jpg')
    _img = cv2.imread('cut/119_231_cut.jpg')

    # _contours = [numpy.array([[100, 180], [200, 280], [200, 180]]), numpy.array([[280, 70], [12, 20], [80, 150]])]
    _contours = [np.array([[200, 280], [100, 180], [200, 180]])]

    fill_255_for_other(_img, _contours, show=True)
