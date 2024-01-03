# from autodiff import *
from Pisor import *
import numpy as np

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import tqdm

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
    
    X_tr = np.array([i[0].numpy() for i in training_data])
    X_te = np.array([i[0].numpy() for i in test_data])
    y_tr = np.array([i[1] for i in training_data])  #(n)
    y_te = np.array([i[1] for i in test_data])
        
    return X_tr, y_tr, X_te, y_te
    
def set_structure(n, k, device = 'cpu'):
    kernel_size1 = 3
    ker1 = pisor(np.random.rand(16, 1, kernel_size1, kernel_size1)/np.sqrt(kernel_size1*kernel_size1),device=device)
    b1 = pisor(np.random.rand(16)/np.sqrt(kernel_size1*kernel_size1),device=device)
    ker2 = pisor(np.random.rand(4, 16, kernel_size1, kernel_size1)/np.sqrt(kernel_size1*kernel_size1),device=device)
    b2 = pisor(np.random.rand(4)/np.sqrt(kernel_size1*kernel_size1),device=device)
    w3 = pisor(np.random.rand(k,4*7*7),device=device)/np.sqrt(4*7*7)
    b3 = pisor(np.random.rand(k)/np.sqrt(k), device=device)

    weights = [ker1, b1, ker2, b2, w3, b3]
    
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
    ker1, b1, ker2, b2, w3, b3 = weights
    X = conv(X, ker1, b1, stride=1, padding=1)  #(n,64,28,28)
    X = pool(X, 2)  #(n,64,14,14)
    # X = relu(X)
    X = conv(X, ker2, b2, stride=1, padding=1)  #(n,16,14,14)
    X = pool(X, 2)  #(n,16,7,7)
    # X = relu(X)
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
     

def opti_epoch(X, y, weights, lr = 1e-5, batch=100, beta1=0.9, beta2=0.999, using_adam=False, device='cpu'):
    if using_adam:
        Adam_epoch(X, y, weights,  lr = lr, batch=batch, beta1=beta1, beta2=beta2, device=device)
    else:
        SGD_epoch(X, y, weights, lr = lr, batch=batch, device=device)

def SGD_epoch(X, y, weights, lr = 0.1, batch=100, lr_decay=0.98, decay_steps=1000, device='cpu'):
    """ 
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
    """
    global t
    
    data_num = X.shape[0]
    for i in range(0,data_num,batch):
        t += 1
        batch_X = pisor.make_const(X[i:i+batch], device=device)
        batch_y = pisor.make_const(y[i:i+batch], device=device)
        pred = forward(batch_X, weights)
        loss = softmax_loss(pred, batch_y)
        loss.backward()
        # print(loss.tensor_number)
        for weight in weights:
            weight.data = weight - lr * weight.grad.detach()   #这步每个运算符都是重载后的
            weight.dirty = False
            weight.grad = None

def Adam_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, lr_decay=0.98, decay_steps=1000, device='cpu'):
    """ 
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum
    """
    global t,ms,vs
    
    data_num = X.shape[0]
    for i in range(0,data_num,batch):
        t += 1
        batch_X = pisor.make_const(X[i:i+batch], device=device)
        batch_y = pisor.make_const(y[i:i+batch], device=device)
        pred = forward(batch_X, weights)
        loss = softmax_loss(pred, batch_y)
        loss.backward()
        for i in range(len(weights)):
            weight = weights[i]
            grad = weight.grad.detach()
            # print(grad.numpy())
            ms[i] = beta1 * ms[i] + (1 - beta1) * grad
            vs[i] = beta2 * vs[i] + (1 - beta2) * grad**2
            m_hat = ms[i] / (1 - beta1**(t))
            v_hat = vs[i] / (1 - beta2**(t))
            update_amount = -lr * m_hat / ((v_hat**0.5) + 1e-8)
            # print(update_amount.numpy())
            weight.data = weight + lr * update_amount
            weight.dirty = False
            weight.grad = None
            

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
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False, device='cpu'):
    
    n, k = X_tr.shape[1], y_tr.max() + 1
    print('start training on '+ device)
    np.random.seed(0)
    weights = set_structure(n, k, device=device)
    #X,y: numpy array
    #weights: list of Tensor
    print("|  Epoch | Test Loss | Test Err |")
    for epoch in tqdm.tqdm(range(epochs)):
        opti_epoch(X_tr, y_tr, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam,device=device)
        # train_loss, train_err = loss_err(forward(pisor.make_const(X_tr,device=device), weights), pisor.make_const(y_tr, device=device))
        test_loss, test_err = loss_err(forward(pisor.make_const(X_te,device=device), weights), pisor.make_const(y_te, device=device))
        # print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
        #       .format(epoch, train_loss[0], train_err, test_loss[0], test_err))
        print("|  {:>4}  |   {:.5f} |  {:.5f} |"\
              .format(epoch, test_loss[0], test_err))


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist() 
    ## using SGD optimizer 
    # train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=128, epochs=400, lr = 8e-5, batch=32, beta1=0.9, beta2=0.999, using_adam=False, device='gpu')
    ## using Adam optimizer
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=128, epochs=200, lr = 1e-2, batch=64, beta1=0.9, beta2=0.999, using_adam=True, device='gpu')
    