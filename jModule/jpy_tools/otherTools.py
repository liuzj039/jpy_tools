'''
@Date: 2020-06-05 22:08:50
LastEditors: liuzj
LastEditTime: 2021-01-29 13:20:18
@Description: 无法归类的工具
@Author: liuzj
FilePath: /jpy_tools/otherTools.py
'''
import os
import sh
import pandas as pd
from loguru import logger
from io import StringIO
import sys
from threading import Thread


class Capturing(list):
    "Capture std output"
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def mkdir(dirPath):
    try:
        sh.mkdir(dirPath)
    except:
        logger.warning(f"{dirPath} existed!!")

        
class Jinterval:
    '''
    自己写的区间操作， 极其不完善
    '''
    def __init__(self, lower, upper, overlapLimit=0.5):
        self.lower, self.upper = lower, upper
        self.interval = [lower, upper]
        self.overlapLimit = overlapLimit

    def __repr__(self):
        return f"Jinterval{self.interval}"

    def __str__(self):
        return f"Jinterval{self.interval}"

    def __and__(self, otherInterval):
        minn = max(self.lower, otherInterval.lower)
        maxn = min(self.upper, otherInterval.upper)
        if (maxn - minn) / (self.upper - self.lower) > self.overlapLimit:
            return [minn, maxn]
        else:
            return False

    def getOverlapRatio(self, otherInterval):
        minn = max(self.lower, otherInterval.lower)
        maxn = min(self.upper, otherInterval.upper)
        return max((maxn - minn) / (self.upper - self.lower), 0)


def creatUnexistedDir(directory):
    '''
    @description: 目录不存在时创建目录
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def isOne(n, i):
    '''
    @description: 判断n的从右向左第i位是否为1
    @param 
        n:{int} 
        i:{int}
    @return: 
        bool
    '''
    return (n & (1 << i)) != 0


def groupby(dtframe, key):
    '''
    @description: 用于groupby操作
    '''
    dtframe.sort_values(key, inplace=True, ignore_index=True)
    dtframeCol = dtframe[key].values
    i = 0
    j = 0
    forwardName = dtframeCol[0]
    for x in range(len(dtframeCol)):
        currentName = dtframeCol[x]
        if currentName == forwardName:
            j += 1
            pass
        else:
            yield dtframe[i:j]
            forwardName = currentName
            i = j
            j += 1
    yield dtframe[i:]

def myAsync(f):
    def wrapper(*args, **kwargs):
        logger.warning("async, you will don't get any results from this function")
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
        return thr
 
    return wrapper