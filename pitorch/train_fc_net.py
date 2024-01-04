from Pisor import *
from optimizer import SGD, Adam 

import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.nn import functional as F
import matplotlib.pyplot as plt
import tqdm
import time

t = 0
ms:list[pisor] = []
vs:list[pisor] = []

def parse_mnist(flatten = True):
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
    
    if(flatten):
        X_tr = np.array([i[0].numpy().reshape(-1) for i in training_data])  #(n,784)
        X_te = np.array([i[0].numpy().reshape(-1) for i in test_data])
    else:
        X_tr = np.array([i[0].numpy()[None] for i in training_data])
        X_te = np.array([i[0].numpy()[None] for i in test_data])
    y_tr = np.array([i[1] for i in training_data])  #(n)
    y_te = np.array([i[1] for i in test_data])
        
    return X_tr, y_tr, X_te, y_te
    
    

def set_structure(n, hidden_dim, k, device = 'cpu'):
    """
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    """
    W1 = pisor(np.random.randn(hidden_dim, n).astype(np.float32) / np.sqrt(hidden_dim), device=device)
    b1 = pisor(np.random.randn(hidden_dim).astype(np.float32) / np.sqrt(hidden_dim), device=device)
    W2 = pisor(np.random.randn(k, hidden_dim).astype(np.float32) / np.sqrt(k), device=device)
    b2 = pisor(np.random.randn(k).astype(np.float32) / np.sqrt(k), device=device)
    weights = [W1,b1, W2,b2]
    
    global t,ms,vs
    t = 0
    ms = [pisor.make_const(np.zeros(weight.numpy().shape), device=device) for weight in weights]
    vs = [pisor.make_const(np.zeros(weight.numpy().shape), device=device) for weight in weights]    
    
    return weights 

def forward(X, weights):
    """
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    """
    X = fc(X, weights[0], weights[1])
    X = relu(X)
    X = fc(X, weights[2], weights[3])
    return X

def softmax_loss(Z:pisor, y:pisor):
    """ 
    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.z
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    batch_size = Z.shape[0]
    
    mask = np.zeros(Z.shape,dtype=int)
    mask[list(range(batch_size)) ,y.numpy().astype(int)] = 1
    mask = raw_pisor(mask, device=Z.device)
    loss = softmax_with_cross_entropy(Z, mask)
    
    return loss    
     

def opti_epoch(X, y, weights, optimizer, batch):
    device = weights[0].device
    data_num = X.shape[0]
    for i in range(0,data_num,batch):
        batch_X = pisor.make_const(X[i:i+batch], device=device)
        batch_y = pisor.make_const(y[i:i+batch], device=device)
        pred = forward(batch_X, weights)  #3.7e-05s
        loss = softmax_loss(pred, batch_y)#5.2e-05s
        loss.backward()                   #8.5e-05s
        optimizer.step()                  #1.9e-04s

def loss_err(h,y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    loss = softmax_loss(h,y).numpy()
    h = h.numpy()
    y = y.numpy()
    error = np.mean(h.argmax(axis=1) != y)
    return loss, error

def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, lr_decay=1., decay_steps=1000, momentum=0.9, beta1=0.9, beta2=0.999, using_adam=False, device='cpu'):
    
    n, k = X_tr.shape[1], y_tr.max() + 1
    print('start training on '+ device)
    np.random.seed(0)
    weights = set_structure(n, k=k, hidden_dim=hidden_dim, device=device)
    if(using_adam):
        optimizer = Adam(weights, lr=lr, lr_decay=lr_decay, decay_steps=decay_steps, beta1=beta1, beta2=beta2)
    else:
        optimizer = SGD(weights, lr=lr, lr_decay=lr_decay, decay_steps=decay_steps, momentum=momentum)
        
    #weights: list of Tensor
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in tqdm.tqdm(range(epochs)):
        opti_epoch(X_tr, y_tr, weights, optimizer, batch=batch)
        train_loss, train_err = loss_err(forward(pisor.make_const(X_tr,device=device), weights), pisor.make_const(y_tr, device=device))
        test_loss, test_err = loss_err(forward(pisor.make_const(X_te,device=device), weights), pisor.make_const(y_te, device=device))
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss[0], train_err, test_loss[0], test_err))


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist() 
    ## using SGD optimizer 
    # train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=128, epochs=400, lr = 1e-2, batch=64, lr_decay=1., decay_steps=10000, momentum=0, using_adam=False, device='gpu')
    ## using Adam optimizer
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=128, epochs=150, lr = 1e-2, batch=64, lr_decay=1., decay_steps=1000, beta1=0.9, beta2=0.999, using_adam=True, device='gpu')
    