import cv2 as cv
import numpy as np


def callback(x):
    """
    空的回调函数，并不需要在这里处理什么
    """
    pass


def setTrackbar():
    """
    创建六个滑动条
    """
    cv.namedWindow("process")
    # 创建滑动条调整参数
    cv.createTrackbar("Rlower", "process", 0, 255, callback)
    cv.createTrackbar("Glower", "process", 0, 255, callback)
    cv.createTrackbar("Blower", "process", 100, 255, callback)
    cv.createTrackbar("Rupper", "process", 72, 255, callback)
    cv.createTrackbar("Gupper", "process", 182, 255, callback)
    cv.createTrackbar("Bupper", "process", 255, 255, callback)


def updateParam():
    """
    返回更新的参数
    """
    r = cv.getTrackbarPos("Rlower", "process")
    g = cv.getTrackbarPos("Glower", "process")
    b = cv.getTrackbarPos("Blower", "process")
    R = cv.getTrackbarPos("Rupper", "process")
    G = cv.getTrackbarPos("Gupper", "process")
    B = cv.getTrackbarPos("Bupper", "process")
    return np.array([b, g, r]), np.array([B, G, R])


class BallDetector:
    """
    检测视频中的小球，默认为检测蓝色小球，阈值可能需要调整
    """

    def __init__(self, index):
        self.index = index
        self.frame = None

    def run(self):
        """
        启动检测小球的主要流程
        """
        try:
            # 打开摄像头或者视频文件
            capture = cv.VideoCapture(self.index)
            # 设置分辨率
            capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            # 设置滑动条
            setTrackbar()
            # 循环执行到视频结束或者摄像头关闭
            while True:
                # 读取一帧
                ret, self.frame = capture.read()
                if self.frame is None:
                    break
                cv.imshow("INPUT", self.frame)
                try:
                    # 主要的检测过程
                    self.process()
                    cv.imshow("OUTPUT", self.frame)
                    cv.waitKey(30)
                except:
                    print("process failed!")
                    cv.waitKey(1000)
            cv.destroyAllWindows()
        except:
            print("open video failed!")

    def process(self):
        """
        对图像进行预处理操作，具体为高斯滤波、阈值化、腐蚀膨胀;
        之后再对预处理好的图像进行轮廓检测操作
        """
        # 高斯滤波后直接拉阈值
        lower, upper = updateParam()
        dst = cv.inRange(cv.GaussianBlur(self.frame, (5, 5), 0), lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        # 先腐蚀，去除很小的杂质
        dst = cv.erode(dst, kernel, iterations=1)
        # 再膨胀，减小反光或者遮挡带来的影响
        dst = cv.dilate(dst, kernel, iterations=3)

        # 显示中间窗口，方便调整阈值
        cv.imshow("process", dst)

        # 轮廓检测
        contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # 检测最小矩阵并且筛选其中接近正方形的一部分
        rects = []
        for i in range(len(contours)):
            x, y, w, h = cv.boundingRect(contours[i])
            if ((h < 30 or w < 30) or
                    abs(h - w) > 20):
                continue
            else:
                rects.append([x, y, w, h])

        # 根据筛选出的矩阵确定圆心和画圆
        for i in range(len(rects)):
            R = (rects[i][2] + rects[i][3]) // 4
            x = rects[i][0] + R
            y = rects[i][1] + R
            cv.circle(self.frame, (x, y), R, (0, 0, 255), 3)


if __name__ == '__main__':
    dec = BallDetector("test.mp4")
    dec.run()

# 总结：
# 之前在基地用C++和OpenCV4写过检测视频中小球的程序，最近在学习Python就试着把之前写的程序简化了一下再转成Python了；
# 感觉OpenCV的Python版本也与C++版本差的很大，python中它可以通过函数返回多个值，并且有一些机制看起来不是很一样，数据格式也变了；
# 比如C++版本的函数常常让你给一个指针进去，把数据保存到指针指向的区域，而python就直接可以从函数返回；
# 感觉自己对python的一些特性还不是很清楚，写程序还不能比较好的发挥这些特性，写起来和C++的区别不是很大。
