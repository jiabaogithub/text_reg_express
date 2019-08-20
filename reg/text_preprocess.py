# coding:UTF-8


import cv2
import numpy as np
from PIL import Image


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst


# 投影法
def projection(imagebin):
    # 再进行图片标准化,将图片数组的数值统一到一定范围内。函数的参数
    # 依次是：输入数组，输出数组，最小值，最大值，标准化模式。
    cv2.normalize(imagebin, imagebin, 0, 255, cv2.NORM_MINMAX)
    import matplotlib.pyplot as plt
    # 使用投影算法对图像投影。
    horizontal_sum = np.sum(imagebin, axis=1)
    print(horizontal_sum)
    plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
    plt.gca().invert_yaxis()
    plt.show()

    # 使用投影算法对图像投影。
    v_sum = np.sum(imagebin, axis=0)
    print(v_sum)
    plt.plot(v_sum, range(v_sum.shape[0]))
    plt.gca().invert_yaxis()
    plt.show()


def detectContours(img):
    # 查找目标轮廓
    # _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape
    # 检测目标轮廓(调试用)
    newCanvas = np.zeros((height, width), np.uint8)
    cv2.drawContours(newCanvas, contours, -1, (255, 255, 255), 1)
    return contours, newCanvas


class PossibleContour(object):
    def __init__(self, contour):
        self.area = cv2.contourArea(contour)
        self.rectX, self.rectY, self.rectWidth, self.rectHeight = cv2.boundingRect(contour)
        self.whratio = self.rectWidth / self.rectHeight
        # compute the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            self.centerX = int(M["m10"] / M["m00"])
            self.centerY = int(M["m01"] / M["m00"])
        else:
            self.centerX, self.centerY = 0, 0


def check_external_contour(contour, height):
    if (
            # contour.area > 0 and contour.rectWidth >= 2 and contour.rectHeight >= height // 3 and contour.rectWidth < contour.rectHeight):
            contour.area > 10 and contour.rectWidth >= 2 and contour.rectHeight >= 2
            # and contour.rectWidth <= contour.rectHeight
    ):
        return True
    else:
        return False


def fix_angle(imagebin, org_img):
    # 计算包含了旋转文本的最小边框
    coords = np.column_stack(np.where(imagebin > 0))
    # print(coords)
    # 该函数给出包含着整个文字区域矩形边框，这个边框的旋转角度和图中文本的旋转角度一致
    angle = cv2.minAreaRect(coords)[-1]
    if abs(int(angle)) == 0:
        return org_img
    # print(angle)
    # 调整角度
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # 仿射变换
    h, w = org_img.shape[:2]
    center = (w // 2, h // 2)
    # print(angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(org_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # cv2.putText(rotated, 'Angle: {:.2f} degrees'.format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # print('[INFO] 旋转角度 :{:.3f}'.format(angle))
    return rotated


def get_contours_fix_angle(imagebin):
    # 检测并绘制目标轮廓
    height, width = imagebin.shape
    image_contour1 = np.zeros((height, width), np.uint8)
    _, contours, hierarchy = cv2.findContours(imagebin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_contour1, contours, -1, (255, 255, 255), 1)
    # cv2.imshow('image_contour1', image_contour1)
    # cv2.waitKey(0)
    # 计算轮廓高度均值
    # height_list = [cv2.boundingRect(contour)[3] for contour in contours]
    # mean_height = np.mean(height_list)
    # 加2是为了保证数字和图片边沿有距离，以便矫正角度
    image_contour2 = np.zeros((height + 2, width + 2), np.uint8)
    for contour in contours:
        contour_possible = PossibleContour(contour)
        if (check_external_contour(contour_possible, height)):
            cv2.rectangle(image_contour2, (contour_possible.rectX, contour_possible.rectY),
                          (contour_possible.rectX + contour_possible.rectWidth,
                           contour_possible.rectY + contour_possible.rectHeight), 255)
    # cv2.imshow('image_contour2', image_contour2)
    # cv2.waitKey(0)
    return image_contour2


def splits_by_contours(imagebin, show=False):
    # 检测并绘制目标轮廓
    height, width = imagebin.shape
    _, contours, hierarchy = cv2.findContours(imagebin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contour1 = np.zeros((height, width), np.uint8)
    cv2.drawContours(image_contour1, contours, -1, (255, 255, 255), 1)

    if show:
        cv2.imshow('image_contour1', image_contour1)
        cv2.waitKey(0)

    # 加100为了给矩形间的缝隙增大1像素距离留出空间
    image_contour2_num_obj = Image.new('RGB', (width + 100, height + 10), "black")
    # image_contour2_num = np.zeros((height, width+100), np.uint8)
    image_contour2_rec = np.zeros((height + 10, width + 100), np.uint8)
    # 过滤不和要求的轮廓
    k = 0
    contours.sort(key=lambda x: cv2.boundingRect(x)[0], reverse=False)
    for i, contour in enumerate(contours):
        contour_possible = PossibleContour(contour)
        if (check_external_contour(contour_possible, height)):
            # 画轮廓的时候将矩形间的缝隙增大1像素距离
            # cv2.drawContours(image_contour2_num, contours, i, (255, 255, 255), 1, offset=(k, 0))
            binary_img_single_num = imagebin[
                                    contour_possible.rectY:contour_possible.rectY + contour_possible.rectHeight,
                                    contour_possible.rectX:contour_possible.rectX + contour_possible.rectWidth]
            binary_img_single_num_obj = Image.fromarray(binary_img_single_num)
            contour_possible.rectX += k
            image_contour2_num_obj.paste(binary_img_single_num_obj, (contour_possible.rectX, contour_possible.rectY))

            cv2.rectangle(image_contour2_rec, (contour_possible.rectX, contour_possible.rectY),
                          (contour_possible.rectX + contour_possible.rectWidth,
                           contour_possible.rectY + contour_possible.rectHeight), 255,thickness=5)
            k += 7
    image_contour2_num = np.array(image_contour2_num_obj)
    # 转单通道
    image_contour2_num = cv2.cvtColor(image_contour2_num, cv2.COLOR_RGB2GRAY)
    if show:
        cv2.imshow('image_contour2_num', image_contour2_num)
        cv2.imshow('image_contour2_rec', image_contour2_rec)
        cv2.waitKey(0)

    # 合并重合的轮廓
    _, contours2, hierarchy = cv2.findContours(image_contour2_rec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contour3 = np.zeros((height + 10, width + 100), np.uint8)
    cv2.drawContours(image_contour3, contours2, -1, (255, 255, 255), 1)
    if show:
        cv2.imshow('image_contour3', image_contour3)
        cv2.waitKey(0)

    contour_filter = []
    for contour in contours2:
        contour_possible = PossibleContour(contour)
        contour_filter.append(contour_possible)

    contour_filter.sort(key=lambda x: x.centerX, reverse=False)
    img_segs = []
    for i, numContour in enumerate(contour_filter):
        img_seg = cv2.getRectSubPix(image_contour2_num, (numContour.rectWidth, numContour.rectHeight),
                                    (numContour.centerX, numContour.centerY))
        # img_seg = cv2.getRectSubPix(imagebin, (numContour.rectWidth, numContour.rectHeight),
        #                             (numContour.centerX, numContour.centerY))
        img_segs.append(img_seg)
        # cv2.imshow("numimage", img_seg);
        # cv2.waitKey(0)
    return img_segs, image_contour3


def divide_obj(regions_obj, num, width_ceiling):
    show = False
    # show = True
    rs = {}
    for img_obj in regions_obj:
        img_y = img_obj[0]
        img = img_obj[1]
        org_img = np.array(img)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)

        # # 二值化
        # imagegray = cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY)
        # # retval, imagebin = cv2.threshold(imagegray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # imagebin = cv2.adaptiveThreshold(imagegray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        #                                  15, 10)
        #
        # # 得到适合水平矫正的二值轮廓图
        # img_of_contours_for_fixangle = get_contours_fix_angle(imagebin)

        # 水平矫正
        # rotated = fix_angle(img_of_contours_for_fixangle, org_img)

        # 伽马变换
        # fgamma = 4
        # image_gamma = np.uint8(np.power((np.array(org_img) / 255.0), fgamma) * 255.0)
        # cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        # cv2.convertScaleAbs(image_gamma, image_gamma)


        # 二值化
        imagegray = cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY)
        imagebin = cv2.adaptiveThreshold(imagegray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 6)
        if show:
            cv2.imshow('imagebin', imagebin)
            cv2.waitKey(0)
        # imagebin = cv2.medianBlur(imagebin, 3)
        # retval, imagebin = cv2.threshold(imagegray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if show:
            cv2.imshow('org_img', org_img)
            # cv2.imshow('image_gamma', image_gamma)
            # cv2.imshow('imagebinmedianBlur', imagebin)
            cv2.waitKey(0)
            # retval, imagebin = cv2.threshold(imagegray_rotated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # cv2.imshow('imagebin', imagebin)
            # cv2.waitKey(0)

        # 图像闭运算
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        # imagebin = cv2.morphologyEx(imagebin, cv2.MORPH_CLOSE, kernel)
        # if show:
        #     cv2.imshow('imagebinCLOSE', imagebin)
        #     cv2.waitKey(0)

        # 利用轮廓分割数字
        imgSegments, imgSegments_contours = splits_by_contours(imagebin, show)

        # 分割图像
        # cols_start, cols_end = verticalDivide_num(imagebin, num)
        # print(cols_start)
        # print(cols_end)
        #
        # # 在矫正后的图上剪切
        # imgSegments = []
        # for i in range(len(cols_start)):
        #     blockSelArr = rotated[:, range(cols_start[i], cols_end[i])]
        #     blockSelArr_bin = imagebin[:, range(cols_start[i], cols_end[i])]
        #
        #     if blockSelArr.shape[1] >= width_ceiling:  # 目标中含有超过该宽度的块，就跳过该目标
        #         # cv2.imshow('blockSelArr', blockSelArr)
        #         # cv2.waitKey(0)
        #         imgSegments = []
        #         break
        #     if blockSelArr_bin.sum() <= 5000 and len(cols_start) > num:  # 小于数字块的最小像素和,且块数大于num，就丢弃
        #         continue
        #
        #     imagegray = cv2.cvtColor(blockSelArr, cv2.COLOR_RGB2GRAY)
        #     # retval, imagebin = cv2.threshold(imagegray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #     # retval, imagebin = cv2.adaptiveThreshold( imagegray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY_INV, 11, 2)
        #     imagebin_ad = cv2.adaptiveThreshold(imagegray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        #                                         15, 10)
        #     imgSegments.append(imagebin_ad)

        # 检查分割情况
        # check_divide(imgSegments)

        if len(imgSegments) == num:  # 过滤出位数是num的目标
            if show:
                cv2.imshow('imagebin', imagebin)
                cv2.imshow('imgSegments_contours', imgSegments_contours)
                cv2.waitKey(0)
                check_divide(imgSegments)
            rs[img_y] = imgSegments
    return rs


def check_divide(imgSegments):
    for i, imgSeg in enumerate(imgSegments):
        pass
        cv2.imshow('imgSeg', imgSeg)
        cv2.waitKey(0)


if __name__ == '__main__':
    imgSegments = [[0, 0, 500], [1, 600, 1], [1000, 1000, 3000], [2000, 2000, 2000], [2000, 1000, 2000],
                   [1000, 2000, 1000], [1000, 3000, 2000]]
    if len(imgSegments) > 4:
        n = len(imgSegments) - 4
        # 计算像素和均值
        pix_sums = [imgSeg.sum() for imgSeg in imgSegments]
        pix_sums_mean = np.mean(pix_sums)
        diff = {i: abs(imgSeg.sum() - pix_sums_mean) for i, imgSeg in enumerate(imgSegments)}
        diff_sorted = dict(sorted(diff.item(), key=lambda x: x[1], reverse=True))[:n + 1]
        new_segs = []
        for i, imgSeg in enumerate(imgSegments):
            if i not in diff_sorted.keys():
                new_segs.append(imgSeg)

'''
        # 锐化
        # blur_img = custom_blur_demo(rotated)
        # 图像闭运算
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # imagebin_ad1 = cv2.morphologyEx(imagebin_ad1, cv2.MORPH_CLOSE, kernel)
        # 腐蚀
        # imagebin_ad1 = cv2.erode(imagebin_ad1, kernel, iterations=1)
        # 膨胀
        # imagebin_ad1 = cv2.dilate(imagebin_ad1, kernel)
'''
