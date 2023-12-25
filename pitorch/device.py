"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们模仿PyTorch定义了一个数据运行框架Device
提供基础的运算接口
"""
import numpy as np


class Device:
    """基类"""


class CPUDevice(Device):
    """CPU Device"""

    def __repr__(self):
        return "cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        return np.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return np.ones(shape, dtype=dtype)

    def randn(self, *shape):
        return np.random.randn(*shape)

    def rand(self, *shape):
        return np.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        return np.eye(n, dtype=dtype)[i]

    def empty(self, shape, dtype="float32"):
        return np.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype="float32"):
        return np.full(shape, fill_value, dtype=dtype)


def cpu():
    return CPUDevice()


def default_device():
    return cpu()


def all_devices():
    return [cpu()]
