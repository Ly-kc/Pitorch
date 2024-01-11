from Pisor import *
from optimizer import SGD, Adam

import numpy as np

from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm
import time

def transform_compose(train=True):
    if(train):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, scale=(0.8,1.2)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

def parse_mnist(flatten = True):
    training_data = datasets.MNIST(
        root='../data',
        train=True,
        download=True,
        transform=transform_compose(train=True)
    )

    test_data = datasets.MNIST(
        root='../data',
        train=False,
        download=True,
        transform=transform_compose(train=False)
    )# ( ((1,28,28), int),....)
    if(flatten):
        X_tr = np.array([i[0].numpy().reshape(-1) for i in training_data])  #(n,784)
        X_te = np.array([i[0].numpy().reshape(-1) for i in test_data])
    else:
        X_tr = np.array([i[0].numpy() for i in training_data])
        X_te = np.array([i[0].numpy() for i in test_data])

    y_tr = np.array([i[1] for i in training_data])  #(n)
    y_te = np.array([i[1] for i in test_data])
        
    return X_tr, y_tr, X_te, y_te
    

def set_structure(n, k, device = 'cpu'):
    kernel_size1 = 3
    ker1 = pisor(
        (np.random.rand(16, 1, kernel_size1, kernel_size1)*2-1) / np.sqrt(kernel_size1*kernel_size1),
        device=device
        )
    b1 = pisor(
        (np.random.rand(16)*2-1) / np.sqrt(kernel_size1*kernel_size1),
        device=device
        )
    ker2 = pisor(
        (np.random.rand(4, 16, kernel_size1, kernel_size1)*2-1) / np.sqrt(16*kernel_size1*kernel_size1),
        device=device
        )
    b2 = pisor(
        (np.random.rand(4)*2-1) / np.sqrt(16*kernel_size1*kernel_size1),
        device=device
        )
    w3 = pisor(
        (np.random.rand(k,4*7*7)*2-1) / np.sqrt(4*7*7),
        device=device)
    b3 = pisor(
        (np.random.rand(k)*2-1) / np.sqrt(4*7*7),
        device=device
        )

    weights = [ker1, b1, ker2, b2, w3, b3]
    
    return weights 

def forward(X, weights):
    """
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    """
    ker1, b1, ker2, b2, w3, b3 = weights
    
    X = conv(X, ker1, b1, stride=1, padding=1)  #(n,64,28,28)
    X = pool(X, 2)  #(n,64,14,14)
    X = relu(X)
    X = conv(X, ker2, b2, stride=1, padding=1)  #(n,16,14,14)
    X = pool(X, 2)  #(n,16,7,7)
    X = relu(X)
    X = X.reshape([X.shape[0], -1])  #(n,16*7*7)
    X = fc(X, w3, b3)  #(n,10)
    
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
        pred = forward(batch_X, weights)
        loss = softmax_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
    # print(optimizer.lr)

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
             epochs=10, lr=0.5, batch=100, lr_decay=0.99, decay_steps=1000, momentum=0.9, beta1=0.9, beta2=0.999, using_adam=False, device='cpu'):
    
    n, k = X_tr.shape[1], y_tr.max() + 1
    print('start training on '+ device)
    np.random.seed(0)
    weights = set_structure(n, k, device=device)
    if(using_adam):
        optimizer = Adam(weights, lr=lr, lr_decay=lr_decay, decay_steps=decay_steps, beta1=beta1, beta2=beta2)
    else:
        optimizer = SGD(weights, lr=lr, lr_decay=lr_decay, decay_steps=decay_steps, momentum=momentum)
    
    test_losses = []
    
    print("|  Epoch | Test Loss | Test Err |")
    for epoch in tqdm.tqdm(range(epochs)):
        opti_epoch(X_tr, y_tr, weights, optimizer, batch)
        test_loss, test_err = loss_err(forward(pisor.make_const(X_te,device=device), weights), pisor.make_const(y_te, device=device))
        print("|  {:>4}  |   {:.5f} |  {:.5f} |"\
              .format(epoch, test_loss[0], test_err))
        test_losses.append(test_loss[0])

    plt.figure()
    plt.plot(test_losses)
    plt.title("Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("test_loss.png")

if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist(flatten=False) 
    ## using SGD optimizer 
    # train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=128, epochs=200, lr=1e-2, lr_decay=0.98, decay_steps=2000, momentum=0.2, batch=64, using_adam=False, device='gpu')
    ## using Adam optimizer
    train_nn(X_tr, y_tr, X_te, y_te, epochs=100, lr = 1e-2, batch=64, lr_decay=0.97, decay_steps=2000, beta1=0.9, beta2=0.999, using_adam=True, device='gpu')
    