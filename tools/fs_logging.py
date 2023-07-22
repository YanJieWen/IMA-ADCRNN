# -*- coding:utf-8 -*-
"""
Software:PyCharm
File: fs_logging.py
Institution: --- CSU&BUCEA ---, China
E-mail: obitowen@csu.edu.cn
Author：Yanjie Wen
Date：2023年04月21日
My zoom: https://github.com/YanJieWen
"""

# 标准的LOGGING模块
import ctypes
import logging
#颜色
FOREGROUND_WHITE = 0x0007
FOREGROUND_BLUE = 0x01  # text color contains blue.
FOREGROUND_GREEN = 0x02  # text color contains green.
FOREGROUND_RED = 0x04  # text color contains red.
FOREGROUND_YELLOW = FOREGROUND_RED | FOREGROUND_GREEN
STD_OUTPUT_HANDLE = -11
# std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

#ctypes记录的颜色方法
# def set_color(color, handle=std_out_handle):
#     bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
#     return bool


class Logger():
    def __init__(self, path, clevel=logging.DEBUG, Flevel=logging.DEBUG):
        '''
        print log
        :param path: file to store
        :param clevel: for cmd
        :param Flevel: for file
        '''
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s]-[%(module)s.py]-[%(lineno)d]-[%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # 设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)

        # 设置文件日志
        fh = logging.FileHandler(path,mode='a')#写入为w，追加为a
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message,color=FOREGROUND_BLUE):
        # set_color(color)
        self.logger.debug(message)
        # set_color(FOREGROUND_WHITE)

    def info(self, message,color=FOREGROUND_GREEN):
        # set_color(color)
        self.logger.info(message)
        # set_color(FOREGROUND_WHITE)

    def war(self, message, color=FOREGROUND_YELLOW):
        set_color(color)
        self.logger.warning(message)
        set_color(FOREGROUND_WHITE)

    def error(self, message, color=FOREGROUND_RED):
        # set_color(color)
        self.logger.error(message)
        # set_color(FOREGROUND_WHITE)

    def cri(self, message):
        self.logger.critical(message)


# if __name__ == '__main__':
#     logyyx = Logger('yyx.log', 'INFO', 'INFO')
#     logyyx.debug('一个debug信息')
#     logyyx.info('一个info信息')
#     logyyx.war('一个warning信息')
#     logyyx.error('一个error信息')
#     logyyx.cri('一个致命critical信息')