"""
本文件我们给出一个基本完善的Tensor类
你可以将lab5的对应代码复制到这里
"""
import numpy as np
from typing import List, Optional, Tuple, Union
from basic_operator import Op, Value
from mytensor import raw_pisor
from mytensor import pEwiseAdd,pAddScalar,pEwiseMul,pMulScalar,pEwiseDiv,pDivScalar,pEwiseSub,pSubScalar
from mytensor import ReLU_forward, ReLU_backward
from autodiff import back_propgation


'''
Tensor(最终要换成Pisor)内含raw_pisor,并在必要时转为numpy
cace_data:raw_pisor
dtype暂时废弃
divice完全存储在raw_pisor中,Tensor不设self.device
'''
class Tensor(Value):
    grad: "Tensor"

    def _init(
        self,
        op,
        inputs,
        *,
        num_outputs: int = 1,
        cached_data: raw_pisor = None,
        requires_grad: Optional[bool] = None
    ):
        self.dirty = False   #是否经历了inplace操作，如果是则不能用来反向传播
        self.grad = None
        super()._init(
            op,
            inputs,
            num_outputs=num_outputs,
            cached_data=cached_data,
            requires_grad=requires_grad,
        )
        
        
    def __init__(
        self,
        array,
        *,
        device: Optional[str] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                #若device和dtype相同则直接引用相同raw_pisor
                cached_data = array.realize_cached_data()  
            else:
                #否则深拷贝生成raw_pisor
                cached_data = Tensor._raw_pisor_from_numpy(
                    array.numpy(), device=device, dtype=dtype  
                )
        elif isinstance(array, np.ndarray):
            if device is None:
                device = 'cpu'
            if dtype is None:
                dtype = array.dtype
            cached_data = Tensor._raw_pisor_from_numpy(array, device=device, dtype=dtype)
        elif(isinstance(array,raw_pisor)):
            cached_data = array
        else:
            if device is None:
                device = 'cpu'
            cached_data = Tensor._raw_pisor_from_array(array, device=device, dtype=dtype)       
            
        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _raw_pisor_from_array(array, device='cpu', dtype=np.float32):
        #array maybe list or tuple
        if(isinstance(array, list) or isinstance(array, tuple)):
            array = np.array(array)
        else:
            array = np.array([array])
        return raw_pisor(np.array(array), device)

    @staticmethod
    def _raw_pisor_from_numpy(numpy_array, device='cpu', dtype=np.float32):
        if(numpy_array.shape == ()):
            numpy_array =numpy_array.reshape(1)
        return raw_pisor(array=numpy_array, device=device)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        if(inputs is not None):
            device = inputs[0].device
            for input in inputs:
                if(input.device != device):
                    raise Exception('parameters not on single device!!!!, please use cpu() or cuda() to change device')
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False, device='cpu'):
        tensor = Tensor.__new__(Tensor)
        if(isinstance(data, np.ndarray)):
            data = raw_pisor(data, device=device)
        elif(isinstance(data, list) or isinstance(data, tuple)):
            data = raw_pisor(np.array(data), device=device)
        elif(isinstance(data, Tensor)):
            data = data.realize_cached_data()
        elif(not isinstance(data, raw_pisor)):
            raise Exception('make_const data type error')
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        '''
        不改变自身的device
        '''
        if(isinstance(value, Tensor)):
            if(self.cached_data.device == 'cpu'):
                self.cached_data = value.realize_cached_data().cpu()
            else:
                self.cached_data = value.realize_cached_data().cuda()
        elif(isinstance(value, np.ndarray)):
            self.cached_data = raw_pisor(value, self.cached_data.device)
        elif(isinstance(value, raw_pisor)):
            if(self.cached_data.device == 'cpu'):
                self.cached_data = value.cpu()
            else:
                self.cached_data = value.gpu()
        else:
            raise Exception('data type error')
        
        self.dirty = True
        

    def detach(self,requies_grad=False):
        return Tensor.make_const(self.realize_cached_data(),requires_grad=requies_grad)

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype()

    @property
    def device(self):
        return self.cached_data.device

    def _cpu(self):
        self.cached_data._cpu()
            
    def _cuda(self):
        self.cached_data._gpu()
    
    def cpu(self):
        return self.detach(requies_grad=self.requires_grad).cpu()
    
    def cuda(self):
        return self.detach(requies_grad=self.requires_grad).cuda()
        
    def partial_gradients(self):
        #输入与返回值均为Tensor类
        return self.op.gradient_as_tuple(self.grad, self)

    def backward(self, out_grad=None):
        back_propgation(self,out_grad)
        

    def __repr__(self):
        return "Tensor(" + str(self.numpy()) + ")"

    def __str__(self):
        return self.numpy().__str__()

    def numpy(self):
        data = self.realize_cached_data().numpy()
        return data


    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return EWisePow()(self, other)
        else:
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def matmul(self, other):
        return MatMul()(self, other)

    def sum(self, axes=None, keep_dims=False):
        return Summation(axes,keep_dims)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def __neg__(self):
        return Negate()(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)

    def inplace_update(self, op, *args):
        self.cached_data = op.compute(self.realize_cached_data(), *args)
        self.dirty = True
        
    def max(self, axis=-1, keep_dims=False):
        return Max(axis,keep_dims)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        raise not NotImplementedError()

class EWiseAdd(TensorOp):
    def compute(self, a: raw_pisor, b: raw_pisor):
        # if(a.device == 'cpu' and b.device == 'cpu'):
        #     return raw_pisor(a.numpy() + b.numpy(), device=a.device)
        # elif(a.device == 'gpu' and b.device == 'gpu'):
        #     return a + b
        # else:
        #     raise Exception('add device error')
        return pEwiseAdd(a,b) 
        
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: raw_pisor):
        # return raw_pisor(a.numpy() + self.scalar)
        return pAddScalar(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: raw_pisor, b: raw_pisor):
        # return raw_pisor(a.numpy() * b.numpy())
        return pEwiseMul(a,b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: raw_pisor):
        # return raw_pisor(a.numpy() * self.scalar)
        return pMulScalar(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: raw_pisor) -> raw_pisor:
        if(a.device == 'cpu'):
            return raw_pisor(a.numpy() ** self.scalar)
        else:
            # raise NotImplementedError()
            print('fake gpu implementation: PowerScalar')
            return raw_pisor(a.numpy() ** self.scalar, 'gpu')
        
    def gradient(self, out_grad, node):
        if(self.scalar == 0):
            return out_grad * 0
        else:
            print(out_grad.device, node.inputs[0].device)
            return out_grad * self.scalar * PowerScalar(self.scalar - 1)(node.inputs[0])
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: raw_pisor, b: raw_pisor) -> np.ndarray:
        if(a.device == 'cpu' and b.device == 'cpu'):
            return raw_pisor(a.numpy()**b.numpy())
        else:
            # raise NotImplementedError()
            print('fake gpu implementation: EWisePow')
            return raw_pisor(a.numpy()**b.numpy(), 'gpu')

    def gradient(self, out_grad, node):
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """逐点相除"""

    def compute(self, a, b):
        # return raw_pisor(np.divide(a.numpy(), b.numpy()))
        return pEwiseDiv(a,b)
        

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs*rhs)
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        if(self.scalar == 0):
            raise ValueError("Divided by zero!")
        # return raw_pisor(a.numpy() / self.scalar)
        return pDivScalar(a, self.scalar)
        
    def gradient(self, out_grad, node):
        return out_grad / self.scalar
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes


    def compute(self, a):
        #助教的意思是axes是一个长为2的tuple，指示着交换哪两个轴？
        #重新翻译为np.transpose的axes参数,即长度为n-1
        if(a.device == 'cpu'):
            new_axes = list(range(len(a.shape)))

            if(self.axes is not None):
                new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
            else:
                new_axes[-1], new_axes[-2] = new_axes[-2], new_axes[-1]
            b = np.transpose(a.numpy(), new_axes)
            b = np.ascontiguousarray(b)
            return raw_pisor(b)
        else:
            # raise NotImplementedError()
            print('fake gpu implementation: Transpose')
            new_axes = list(range(len(a.shape)))

            if(self.axes is not None):
                new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
            else:
                new_axes[-1], new_axes[-2] = new_axes[-2], new_axes[-1]
            b = np.transpose(a.numpy(), new_axes)
            b = np.ascontiguousarray(b)
            return raw_pisor(b, 'gpu')         
    
    def gradient(self, out_grad, node):
        #这里axes为None以及为（x,y）的情况都考虑了
        return transpose(out_grad, self.axes)
    
    ##基于numpy关于axes约定的实现
    # def gradient(self, out_grad, node):
    #     reverse_axes = None
    #     if(self.axes is not None):
    #         natural = np.arange(len(self.axes)) 
    #         reverse_axes = np.zeros_like(natural)
    #         reverse_axes[np.array(self.axes)] = natural
            
    #     return transpose(out_grad, reverse_axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)
        

    def gradient(self, out_grad, node):
        return Reshape(node.inputs[0].shape)(out_grad)
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
    
    #numpy不支持(3,4)->(6,4)  
    #在此暂时只实现(3,1,4)->(8,3,3,4)的情况
    def compute(self, a):
        if(a.device == 'cpu'):
            res = np.zeros(self.shape)
            res += a.numpy()
            return raw_pisor(res)
        else:
            # raise NotImplementedError()
            print('fake gpu implementation: BroadcastTo')
            res = np.zeros(self.shape)
            res += a.numpy()
            return raw_pisor(res,'gpu')
                
    def gradient(self, out_grad, node):
        origin_shape = node.inputs[0].shape
        extra_axis_num = len(self.shape) - len(origin_shape)
        compress_axis = list(range(extra_axis_num))
        for i in range(len(origin_shape)):
            if(origin_shape[i] != self.shape[i + extra_axis_num]):
                compress_axis.append(i + extra_axis_num)   

        res = summation(out_grad, axes=tuple(compress_axis))
        if(len(res.shape) != len(origin_shape)):
            res = res.reshape(origin_shape)
        return res


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keep_dims = False):
        self.axes = axes
        self.keep_dims = keep_dims

    def compute(self, a):
        if(a.device == 'cpu'):
            return raw_pisor(np.sum(a.numpy(), axis=self.axes,keepdims=self.keep_dims))
        else:
            # raise NotImplementedError()
            print('fake gpu implementation: Summation')
            return raw_pisor(np.sum(a.numpy(), axis=self.axes,keepdims=self.keep_dims), 'gpu')
        
    def gradient(self, out_grad, node):
        #烦人的零维Tensor
        if(len(node.inputs[0].shape) == 0 and len(out_grad.shape) == 0):
            return out_grad
        if(self.keep_dims == False):
            if(node.inputs[0].shape == ()):
                whole_shape = np.array([1])
            else:
                whole_shape = np.zeros(len(node.inputs[0].shape),dtype=int)
            
            if(self.axes is not None):
                whole_shape[np.array(self.axes)] = 1
            else:
                whole_shape[:] = 1
            whole_shape[whole_shape == 0] = np.array(out_grad.shape)

            unsqueezed_out_grad = reshape(out_grad, tuple(whole_shape))
            return broadcast_to(unsqueezed_out_grad, node.inputs[0].shape)
        else:
            return broadcast_to(out_grad, node.inputs[0].shape)
        


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        if(a.device == 'cpu'):
            return raw_pisor(np.matmul(a.numpy(), b.numpy()))
        else:
            print('fake gpu implementation: MatMul')
            return raw_pisor(np.matmul(a.numpy(), b.numpy()), 'gpu')
        
    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad @ transpose(rhs), transpose(lhs) @ out_grad
        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return pMulScalar(a, -1)

    def gradient(self, out_grad, node):
        return -out_grad
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        if(a.device == 'cpu'):
            return raw_pisor(np.log(a.numpy()))
        else:
            print('fake gpu implementation: Log')
            return raw_pisor(np.log(a.numpy()), 'gpu')
                
    def gradient(self, out_grad, node):
        return out_grad/node.inputs[0]
        
def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        if(a.device == 'cpu'):
            return raw_pisor(np.exp(a.numpy()))        
        else:
            print('fake gpu implementation: Exp')
            return raw_pisor(np.exp(a.numpy()), 'gpu')
        
    def gradient(self, out_grad, node):
        return out_grad * Exp()(node)
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # return raw_pisor(np.maximum(a.numpy(), 0))
        return ReLU_forward(a)
        
    def gradient(self, out_grad, node):
        # #生成一个mask
        # mask = node.inputs[0].numpy() > 0
        # mask = Tensor.make_const(mask) #不需要关于mask的梯度
        # return out_grad * mask
        return Tensor(ReLU_backward(node.inputs[0].realize_cached_data(), out_grad.realize_cached_data()))
        
def relu(a):
    return ReLU()(a)


#暂时只支持索引最后一维
class Index(TensorOp):
    def __init__(self, index):
        self.index = index  
    
    def compute(self, a):
    #(...,n,c)->(...,n,)
        if(a.device == 'cpu'):
            return raw_pisor(a.numpy()[...,self.index])
        else:
            print('fake gpu implementation: Index')
            return raw_pisor(a.numpy()[...,self.index], 'gpu')
        
    def gradient(self, out_grad, node):
    #(...,n,)->(...,n,c)
        mask = np.zeros_like(node.inputs[0].numpy())
        mask[...,self.index] = 1
        mask = Tensor.make_const(mask)
        full_out_grad = broadcast_to(out_grad, node.inputs[0].shape)
        return full_out_grad * mask

def op_index(a, index):
    return Index(index)(a)



class Assign_mask(TensorOp):
    def __init__(self, mask:np.ndarray):
        self.mask = mask
    
    def compute(self, a):
        # return raw_pisor(a.numpy() * self.mask)
        mask = raw_pisor(self.mask, device = a.device)
        return pEwiseMul(a,mask)
            
    def gradient(self, out_grad, node):
        return assign_mask(out_grad, self.mask)
    
def assign_mask(a, mask):
    return Assign_mask(mask)(a)



class Max(TensorOp):
    def __init__(self, axis = -1,keep_dims=False):
        self.axis = axis
        self.keep_dims = keep_dims
    
    def compute(self, a):
        if(a.device == 'cpu'):
            return raw_pisor(np.max(a.numpy(), axis=self.axis,keepdims=self.keep_dims))
        else:
            print('fake gpu implementation: Max')
            return raw_pisor(np.max(a.numpy(), axis=self.axis,keepdims=self.keep_dims), 'gpu')
            
    def gradient(self, out_grad, node):
        #生成一个mask
        mask = node.inputs[0].numpy() == node.numpy()
        mask = Tensor.make_const(mask,device=out_grad.device) #不需要关于mask的梯度
        if(self.keep_dims == False):
            if(node.inputs[0].shape == ()):
                whole_shape = np.array([1])
            else:
                whole_shape = np.zeros(len(node.inputs[0].shape),dtype=int)
            
            if(self.axis is not None):
                whole_shape[self.axis] = 1
            else:
                whole_shape[:] = 1
            whole_shape[whole_shape == 0] = np.array(out_grad.shape)

            unsqueezed_out_grad = reshape(out_grad, tuple(whole_shape))
            return broadcast_to(unsqueezed_out_grad, node.inputs[0].shape) * mask
        
        else:
            #out_grad:(n,1)  mask:(n,c)
            unsqueezed_out_grad = broadcast_to(out_grad, node.inputs[0].shape)  #(n,c)
            return unsqueezed_out_grad * mask
    
