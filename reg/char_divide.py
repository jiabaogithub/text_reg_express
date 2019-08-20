# coding:UTF-8


# 水平切割掉图片上下两侧不感兴趣的区域


def horizontalDivide(imArr):
    h = imArr.shape[0]
    w = imArr.shape[1]
    # 水平切割并返回切割的坐标
    a = 0
    b = 0
    af = 0
    bf = 0
    len = int(imArr.shape[0] / 2) - 1  # 中间行号
    for i in reversed(range(len)):
        if w - imArr[i, :].sum(axis=0) >= 480:
            a = i
            break
    # for i in range(len+1,h-1):
    #     if w -imArr[i, :].sum(axis=0) >= 300:
    #         b = i-1
    #         break
    # imArr = imArr[range(a, b), :]
    imArr = imArr[range(a, len + 35), :]
    imArr[imArr[:, :] == 1] = 255
    return imArr


# 水平切割并返回分割图片特征
def horizontalDivide(imArr):
    c = []
    d = []
    imgSegments = []
    height = imArr.shape[0] - 1
    for j in range(height):
        if (imArr[j, :] == 0).all() and (imArr[j + 1, :] == 255).any():
            c.append(j + 1)
        elif (imArr[j, :] == 255).any() and (imArr[j + 1, :] == 0).all():
            d.append(j + 1)
    # if len(c) != 0 and len(d) != 0 and len(c) == len(d):
    #     for i in range(len(c)):
    #         blockSelArr = imArr[range(c[i], d[i]), :]
    #         imgSegments.append(blockSelArr)
    return imgSegments, c, d


# 垂直切割并返回分割图片特征
def verticalDivide(imArr):
    c = []
    d = []
    imgSegments = []
    width = imArr.shape[1] - 1
    for j in range(width):
        if (imArr[:, j] == 0).all() and (imArr[:, j + 1] == 255).any():
            c.append(j + 1)
        elif (imArr[:, j] == 255).any() and (imArr[:, j + 1] == 0).all():
            d.append(j + 1)
    # if len(c) != 0 and len(d) != 0 and len(c) == len(d):
    #     for i in range(len(c)):
    #         blockSelArr = imArr[:, range(c[i], d[i])]
    #         imgSegments.append(blockSelArr)
    return imgSegments, c, d


# 垂直切割并返回分割图片特征
def verticalDivide_zhw(imArr, char_min_with):
    c = []
    d = []
    jump = False
    imgSegments = []
    width = imArr.shape[1] - 1
    for j in range(width):
        if (imArr[:, j] == 0).all() and (imArr[:, j + 1] == 255).any() and not jump:
            c.append(j + 1)
        elif (imArr[:, j] == 255).any() and (imArr[:, j + 1] == 0).all():
            if (j + 1 - c[len(c) - 1]) >= char_min_with:
                d.append(j + 1)
                jump = False
            else:  # 遇到了左右结构的汉字，此时左边部分的结束位置(即当前结束位置)和右边部分的开始位置（即下一个开始位置）都要跳过
                jump = True
    for i in range(len(c)):
        blockSelArr = imArr[:, range(c[i], d[i])]
        imgSegments.append(blockSelArr)
    return imgSegments, c, d


# 垂直切割并返回分割图片特征
def verticalDivide_num(imArr, num):
    c = []
    d = []
    s = 0
    # imArr = imArr[0:-5,]
    width = imArr.shape[1]
    char_min_with = width // num * 2
    # char_min_with = width//2 # 如此将先屏蔽此条件

    if (imArr[:, 0] == 255).any():
        c.append(0)
        for j in range(width):
            if (imArr[:, j] == 0).all():
                d.append(j)
                break
        if len(d) == 0:  # 说明一直到最后都没有全黑色的分隔
            d.append(width - 1)
        s = d[0]
    for j in range(s, width):
        if j == 35:
            print(j)
        if j == width - 1:
            if len(c) > len(d):
                d.append(j)
            elif len(c) < len(d):
                c.append(j)
            break
        if (imArr[:, j] == 0).all() and (imArr[:, j + 1] == 255).any():
            c.append(j)
        elif (imArr[:, j] == 255).any() and (imArr[:, j + 1] == 0).all():
            if (j + 1 - c[len(c) - 1]) >= char_min_with:  # 遇到了连在一起的数字，此时要进行切分
                d.append(j + 1 - (j + 1 - c[len(c) - 1]) // 2 + 1)
                c.append(j + 1 - (j + 1 - c[len(c) - 1]) // 2 + 1 + 1)
                d.append(j + 1)
            else:
                d.append(j + 1)
        elif (imArr[:, j] == 255).any() and j == width - 1:
            if (j - c[len(c) - 1]) >= char_min_with:  # 遇到了连在一起的数字，此时要进行切分
                d.append(j - (j - c[len(c) - 1]) // 2 + 1)
                c.append(j - (j - c[len(c) - 1]) // 2 + 1 + 1)
                d.append(j)
            else:
                d.append(j)

    return c, d


# 二值处理
def convertToTwoValueImg(imgArray):
    # 转化为二值矩阵，0以外的值全部转为255
    for i in range(imgArray.shape[0]):
        for j in range(imgArray.shape[1]):
            if (imgArray[i, j] > 70).any():
                imgArray[i, j] = 1
            else:
                imgArray[i, j] = 0
    return imgArray
