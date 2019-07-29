import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def error_poly(C, data):
    '''
    计算给定系数的多项式函数和给定数据的误差平方差
    :param C: numpy.poly1d 类型数据或者数组类型的多项式系数
    :param data: 2D 数组数据
    :return:
    '''
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
    return err


def fit_poly(data, error_func, degree=3):
    '''
    通过给定数据和err_func计算拟合的多项式
    :param data:
    :param error_func:
    :param degree:
    :return:
    '''
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

    x = np.linspace(-5, 5, 21)
    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="initial guess")

    result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'disp': True})
    return np.poly1d(result.x)

