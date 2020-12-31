'''
@Date: 2020-06-05 22:01:45
@LastEditors: liuzj
@LastEditTime: 2020-06-05 22:04:13
@Description: 主要用于多进程处理
@Author: liuzj
@FilePath: /liuzj/softwares/python_scripts/jpy_modules/jpy_tools/MultiProcess.py
'''
import os
import numpy as np
import pandas as pd


def multfunc_dtframe(func,data,threading,use_iter=False,use_threading=False,*args):
    if use_threading:
        from multiprocessing.dummy import Pool
    else:
        from multiprocessing import Pool
    result=[]
    if not use_iter:
        pool=Pool(threading)
        span=(len(data)//threading)+1
        for _ in range(threading):
            sub_data=data.iloc[_*span:(_+1)*span]
            result.append(pool.apply_async(func,args=(sub_data,*args)))
        pool.close()
        pool.join()
        result=[x.get() for x in result]
    else:
        forward_chunk_result=[]
        latter_chunk_result=[]
        chunk_data=next(data)
        while True:

            pool=Pool(threading)
            if chunk_data.empty:
                break
            else:
                span=(len(chunk_data)//(threading-1))+1
                for _ in range(threading-1):
                    sub_data=chunk_data.iloc[_*span:(_+1)*span]
                    if not sub_data.empty:
                        latter_chunk_result.append(pool.apply_async(func,args=(sub_data,*args)))
                try:
                    chunk_data=next(data)
                    if forward_chunk_result:
                        forward_chunk_result=[x.get() for x in forward_chunk_result]
                        result.extend(forward_chunk_result)
                    pool.close()
                    pool.join()
                    forward_chunk_result=latter_chunk_result
                    latter_chunk_result=[]
                except:
                    if forward_chunk_result:
                        forward_chunk_result=[x.get() for x in forward_chunk_result]
                        result.extend(forward_chunk_result)
                    pool.close()
                    pool.join()
                    forward_chunk_result=latter_chunk_result
                    latter_chunk_result=[]
                    break
        forward_chunk_result=[x.get() for x in forward_chunk_result]
        result.extend(forward_chunk_result)
    
    return result


def ufunc_dec(func,dtframe,index,output_count,*args):
    dtframe=dtframe.values
    func=np.frompyfunc(func,len(index)+len(args),output_count)
    order=''
    _=''
    for i in index:
        _ += 'dtframe[:,%d],'%i
    for i in args:
        _ += '%s,'%i
    _=_[:-1]
    order='func(%s)'%(_)
    result_dt=eval(order)
    return result_dt


def _singleApplyFunc(subDtframe, func):
    subResults = subDtframe.apply(func, axis=1)
    return subResults


def multiApplyFunc(allDtframe,func,threads):
    '''
    @description: 用于多进程处理dataframe
    @param: 
        allDtframe: 需要处理的数据。
        func: 一个正常的apply函数。 
    @return: 
        类似apply返回的结果
    '''
    allResults = multfunc_dtframe(_singleApplyFunc, allDtframe, threads, False, False, func)
    allResults = pd.concat(allResults)
    return allResults