# VerificationMatcher
记一次使用openCV+tesseract对验证码识别过程


# 目标介绍

验证码来自某招聘网站企业版[登录入口](https://passport.zhaopin.com/org/login)

大概长这样:![](http://owu391pls.bkt.clouddn.com/yanzhenma.png)

我把上面的图叫做src，下面的图叫做query，要做的就是根据src，找到query中相应的位置
src:![](http://owu391pls.bkt.clouddn.com/src1.png)

query:![](http://owu391pls.bkt.clouddn.com/query1.png)



一些方法的封装：

```python
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
```





# 模板匹配

多刷新几次你会发现 src的文字在图片中的位置都是固定的

那处理起来就很方便了，灰度化之后直接按像素分割

```python
src1 = cv2.imread(resPath + '/src1.png', cv2.COLOR_BGR2BGRA)
words = (src1[0:h, 10:32], src1[0:h, 30:53], src1[0:h, 51:73])
```

query更简单，灰度化处理一下

```python
query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
```

然后就可以开始匹配了

![](https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=517938955,2498068913&fm=27&gp=0.jpg)



```python
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
```

结果:![](http://owu391pls.bkt.clouddn.com/result1.png)





上面写的也太水了吧，没有困难我们就创造困难。。比如人家把字体大小、分辨率给改了，那么模板匹配很可能就不行了

# OCR文字识别

大体思路就是分割src，通过谷歌的tesseract转换成文字，再把query中的文字一个个抓出来识别，因为query中字体位置无规律，一整张放进去也无法识别。

src:![](http://owu391pls.bkt.clouddn.com/src2.png)

src的处理就不多BB了，二值化之后固定分割，甚至可以不处理直接分割，然后再识别即可主要看query的处理。

query:![](http://owu391pls.bkt.clouddn.com/query2.png)

二值化：

```python
    gray_query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    binary_query = binary_filter(gray_query, thresh=30)
```

![](http://owu391pls.bkt.clouddn.com/query_binary.png)

由于字体是黑色的，所以我们要通过腐蚀操作处理一下（其实可以先膨胀去除右边的小黑点再腐蚀的），让下面轮廓识别提取坐标更准确一些：

```python
	erode_query = erode_img(binary_query)
```



![](http://owu391pls.bkt.clouddn.com/query_erode.png)

注意到轮廓识别是在黑色背景中，识别白色轮廓，所以要做个颜色翻转，再识别：

```python
    # 翻转颜色来查找轮廓
    erode_query = cv2.bitwise_not(erode_query)
    image, contours, hierarchy = cv2.findContours(erode_query, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓保存 以便观察
    query = cv2.drawContours(query, contours, -1, (0, 0, 255), 1)
    cv2.imwrite(resPath + '/query_contours.png', query)
```

在原图上把轮廓画出来，如下

![](http://owu391pls.bkt.clouddn.com/query_contours.png)



然后处理一下轮廓数据，去掉一些轮廓siez明显不符合的，以及把轮廓整成矩形的

```python
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
```

有了每个文字的坐标，就可以抠出这些文字来进行ocr了

```python
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
```

识别出来的word如下：

![](http://owu391pls.bkt.clouddn.com/query_word0.png)
![](http://owu391pls.bkt.clouddn.com/query_word1.png)
![](http://owu391pls.bkt.clouddn.com/query_word2.png)
![](http://owu391pls.bkt.clouddn.com/query_word3.png)
![](http://owu391pls.bkt.clouddn.com/query_word4.png)
![](http://owu391pls.bkt.clouddn.com/query_word5.png)
![](http://owu391pls.bkt.clouddn.com/query_word6.png)

第一个看来轮廓没处理好，但是也不碍事～