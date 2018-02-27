# coding=utf-8
import os
import cv2
import numpy as np
import pytesseract


# 二值化
def binary_filter(src, thresh=127, maxval=255):
    ret1, th1 = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)
    return th1


# 腐蚀
def erode_img(src, shape=(5, 5)):
    kernel = np.ones(shape, np.uint8)
    return cv2.erode(src, kernel, iterations=1)


# 膨胀
def dilate_img(src):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(src, kernel, iterations=1)


# ocr
def img_to_cn(wordImg):
    return pytesseract.image_to_string(wordImg, lang='my_cn')


# rgb三色各表示123文字点击顺序
# src文字分割直接写死坐标
def template_match(src, query):
    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    w, h = src.shape[::-1]
    words = (src[0:h, 10:32], src[0:h, 30:53], src[0:h, 51:73])
    for i in range(len(words)):
        template = words[i]
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(query_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(query, pt, (pt[0] + w, pt[1] + h), colors[i], 2)
    cv2.imwrite(resPath + '/result1.png', query)


# 切割文字进行ocr识别比对
def ocr_match(src, query):
    # 宽度固定 直接对src进行二值化后切割
    h = src.shape[0]
    src = binary_filter(src, thresh=25)
    words = (src[0:h, 10:32], src[0:h, 30:53], src[0:h, 51:73])
    # 对src来说字体分布不规则 则需要二值化后经过腐蚀和颜色翻转，然后通过轮廓识别来确定每个文字位置
    gray_query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    binary_query = binary_filter(gray_query, thresh=30)
    # 保存处理后的图片 以便观察
    cv2.imwrite(resPath + '/query_binary.png', binary_query)
    erode_query = erode_img(binary_query)
    # 保存处理后的图片 以便观察
    cv2.imwrite(resPath + '/query_erode.png', erode_query)
    # 翻转颜色来查找轮廓
    erode_query = cv2.bitwise_not(erode_query)
    image, contours, hierarchy = cv2.findContours(erode_query, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓保存 以便观察
    query = cv2.drawContours(query, contours, -1, (0, 0, 255), 1)
    cv2.imwrite(resPath + '/query_contours.png', query)
    # 过滤并计算轮廓的最大值 最小值 以矩形切割字符
    coorList = []
    for i in range(0, len(contours), 1):
        ndarr = contours[i]
        # 根据轮廓大小过滤
        if ndarr.size < 30:
            continue
        maxX = 0
        minX = 1000  # h=src.shape[1]
        maxY = 0
        minY = 1000  # w=src.shape[0]
        for j in range(0, ndarr.shape[0], 1):
            ndarr2 = ndarr[j]
            ndarr3 = ndarr2[0]
            if ndarr3[0] > maxX:
                maxX = ndarr3[0]
            if ndarr3[0] < minX:
                minX = ndarr3[0]
            if ndarr3[1] > maxY:
                maxY = ndarr3[1]
            if ndarr3[1] < minY:
                minY = ndarr3[1]
        coorList.append((minX, maxX, minY, maxY))
    # 根据形状来切割每个字符 并进行ocr识别
    for i in range(0, len(coorList), 1):
        coorTuple = coorList[i]
        wordImg = binary_query[coorTuple[2]:coorTuple[3], coorTuple[0]:coorTuple[1]]
        # 保存每个字符 便于观察
        cv2.imwrite(resPath + '/query_word{idx}.png'.format(idx=i), wordImg)
        # ocr匹配
        if img_to_cn(wordImg) == img_to_cn(words[0]):
            print('第一个点击位置为(%d,%d)', (sum(coorTuple[:2]) / 2), (sum(coorTuple[2:]) / 2))
        elif img_to_cn(wordImg) == img_to_cn(words[1]):
            print('第二个点击位置为(%d,%d)', (sum(coorTuple[:2]) / 2), (sum(coorTuple[2:]) / 2))
        elif img_to_cn(wordImg) == img_to_cn(words[2]):
            print('第三个点击位置为(%d,%d)', (sum(coorTuple[:2]) / 2), (sum(coorTuple[2:]) / 2))


resPath = os.path.dirname(os.path.realpath(__file__)) + '/res'

src1 = cv2.imread(resPath + '/src1.png', cv2.COLOR_BGR2BGRA)
query1 = cv2.imread(resPath + '/query1.png')
template_match(src1, query1)

src2 = cv2.imread(resPath + '/src2.png', cv2.COLOR_BGR2BGRA)
query2 = cv2.imread(resPath + '/query2.png')
ocr_match(src2, query2)
