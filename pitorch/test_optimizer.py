"""
本文件我们尝试实现一个Optimizer类，用于优化一个简单的双层Linear Network
本次作业主要的内容将会在opti_epoch内对于一个epoch的参数进行优化
分为SGD_epoch和Adam_epoch两个函数，分别对应SGD和Adam两种优化器
其余函数为辅助函数，也请一并填写
和大作业的要求一致，我们不对数据处理和读取做任何要求
因此你可以引入任何的库来帮你进行数据处理和读取
理论上我们也不需要依赖lab5的内容，如果你需要的话，你可以将lab5对应代码copy到对应位置
"""
from autodiff import *
from Pisor import *
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.nn import functional as F
import matplotlib.pyplot as plt
import tqdm

t = 0
ms:list[np.ndarray] = []
vs:list[np.ndarray] = []

def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    training_data = datasets.MNIST(
        root='../data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root='../data',
        train=False,
        download=True,
        transform=ToTensor()
    )# ( ((1,28,28), int),....)
    
    X_tr = np.array([i[0].numpy().reshape(-1) for i in training_data])  #(n,784)
    y_tr = np.array([i[1] for i in training_data])  #(n)
    X_te = np.array([i[0].numpy().reshape(-1) for i in test_data])
    y_te = np.array([i[1] for i in test_data])
    
    return X_tr, y_tr, X_te, y_te
    
    

def set_structure(n, hidden_dim, k, device = 'cpu'):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    Example:
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return list(W1, W2)
    """
    # k = k.realize_cached_data()
    W1 = pisor(np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim), device=device)
    W2 = pisor(np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k), device=device)
    W_res = pisor(np.random.randn(n, k).astype(np.float32) / np.sqrt(k), device=device)
    # weights = [W1, W2, W_res]
    weights = [W1, W2]
    
    global t,ms,vs
    t = 0
    ms = [np.zeros(weight.numpy().shape) for weight in weights]
    vs = [np.zeros(weight.numpy().shape) for weight in weights]    
    
    return weights 

def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    Example:
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2
    """
    W1 = weights[0]   
    W2 = weights[1]
    # W_res = weights[2]
    
    # output1 = relu(X@W1)@W2
    # output2 = X@W_res
    # return output1 + output2
    
    return relu(X@W1)@W2

def softmax_loss(Z:pisor, y:pisor):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.z
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    #如何仅凭手上有的算符计算softmax loss。。
    #需要在Tensor里加入一个index算子，然后模仿下面的代码
    #return -np.mean(np.log(np.exp(Z[np.arange(len(y)),y]) / np.sum(np.exp(Z), axis=1)))
    # 先搞个（n，c）的mask（无需梯度），与z相乘，再sum最后一维得到（n，）
    batch_size = Z.shape[0]
    #防止溢出，首先减去最大值
    maxz = Z.max(axis=-1,keep_dims=True)  #(n,1)
    maxz = broadcast_to(maxz, Z.shape)  #(n,c)
    Z = Z - maxz
    
    mask = np.zeros(Z.shape,dtype=int)
    mask[list(range(batch_size)) ,y.numpy().astype(int)] = 1
    masked_z = assign_mask(Z,mask) # (n,c)
    zy = masked_z.sum(axes=-1)  # (n,)  
    
    sum_ez = exp(Z).sum(axes=-1)  # (n,)
    ey = exp(zy)  # (n,)

    loss = log(sum_ez/ey).sum(-1) / batch_size
    # mask = np.zeros(Z.shape,dtype=int)
    # mask[list(range(batch_size)) ,y.realize_cached_data()] = 1
    # masked_z = assign_mask(Z,mask) # (n,c)
    # zy = masked_z.sum(axes=-1,keep_dims=True)  # (n,1)
    # print(zy.numpy()[0:10,0])
    # zy = broadcast_to(zy, masked_z.shape) # (n,c)
    # z_dis_exp = exp(Z - zy) # (n,c)
    # z_dis_exp_sum = z_dis_exp.sum(axes=-1) # (n,)
    # loss = log(z_dis_exp_sum).sum(-1) / batch_size
    
    return loss    
     

def opti_epoch(X, y, weights, lr = 1e-5, batch=100, beta1=0.9, beta2=0.999, using_adam=False, device='cpu'):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    if using_adam:
        Adam_epoch(X, y, weights,  lr = lr, batch=batch, beta1=beta1, beta2=beta2, device=device)
    else:
        SGD_epoch(X, y, weights, lr = lr, batch=batch, device=device)

def SGD_epoch(X, y, weights, lr = 0.1, batch=100, device='cpu'):
    """ 
    SGD优化一个List of Weights
    本函数应该inplace地修改Weights矩阵来进行优化
    用学习率简单更新Weights

    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
    """
    data_num = X.shape[0]
    for i in range(0,data_num,batch):
        batch_X = pisor.make_const(X[i:i+batch], device=device)
        batch_y = pisor.make_const(y[i:i+batch], device=device)
        pred = forward(batch_X, weights)
        # print(pred.numpy()[0])
        loss = softmax_loss(pred, batch_y)
        loss.backward()
        for weight in weights:
            # weight.inplace_update(EWiseAdd(), -lr * weight.grad.realize_cached_data())
            weight.data = weight - lr * weight.grad   #这步每个运算符都是重载后的
            weight.dirty = False
            weight.grad = None
        # print(loss.realize_cached_data())
    

def Adam_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999,device='cpu'):
    """ 
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum

    Returns:
        None
    """
    global t,ms,vs
    
    data_num = X.shape[0]
    for i in range(0,data_num,batch):
        t += 1
        batch_X = pisor.make_const(X[i:i+batch],device=device)
        batch_y = pisor.make_const(y[i:i+batch],device=device)
        pred = forward(batch_X, weights)
        # print(pred.numpy()[0])
        loss = softmax_loss(pred, batch_y)
        loss.backward()
        for i in range(len(weights)):
            weight = weights[i]
            grad = weight.grad.realize_cached_data()
            ms[i] = beta1 * ms[i] + (1 - beta1) * grad  
            vs[i] = beta2 * vs[i] + (1 - beta2) * grad**2
            m_hat = ms[i] / (1 - beta1**(t))
            v_hat = vs[i] / (1 - beta2**(t))
            update_amount=  -lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            weight.inplace_update(EWiseAdd(), update_amount)
            weight.dirty = False
            weight.grad = None
        # print(loss.realize_cached_data())

    return t

def loss_err(h,y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    loss = softmax_loss(h,y).numpy()
    h = h.numpy()
    y = y.numpy()
    error = np.mean(h.argmax(axis=1) != y)
    return loss, error

def train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False, device='cpu'):
    """ 
    训练过程
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    weights = set_structure(n, hidden_dim, k, device=device)
    np.random.seed(0)
    
    #X,y: numpy array
    #weights: list of Tensor
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        opti_epoch(X_tr, y_tr, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam)
        train_loss, train_err = loss_err(forward(pisor.make_const(X_tr), weights), pisor.make_const(y_tr))
        test_loss, test_err = loss_err(forward(pisor.make_const(X_te), weights), pisor.make_const(y_te))
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist() 
    weights = set_structure(X_tr.shape[1], 100, y_tr.max() + 1)
    ## using SGD optimizer 
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=350, lr = 5e-6, batch=64, beta1=0.9, beta2=0.999, using_adam=False)
    ## using Adam optimizer
    # train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=128, epochs=200, lr = 1e-4, batch=64, beta1=0.9, beta2=0.999, using_adam=True)
    