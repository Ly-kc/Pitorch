from Pisor import *

class optimizer():
    def __init__(self, weights:[pisor], lr, lr_decay=1, decay_steps=1000):
        self.t = 0
        self.weights = weights
        self.lr = lr
        self.lr_decay = lr_decay
        self.decay_steps = decay_steps
        self.device = weights[0].device

    def step(self):
        raise NotImplementedError()
    
    def zero_grad(self):
        for weight in self.weights:
            weight.grad = None
    

class SGD(optimizer):
    def __init__(self, weights:[pisor], lr=0.1, lr_decay=1, decay_steps=1000, momentum=0.9):
        super().__init__(weights, lr, lr_decay, decay_steps)
        self.vel:[pisor] = []
        self.momentum = momentum

    def step(self):
        self.t += 1

        for i,weight in enumerate(self.weights):
            if(self.t == 1):
                self.vel.append(weight.grad.detach())
            else:
                self.vel[i] = self.momentum*self.vel[i] + (1-self.momentum)*weight.grad.detach()
            weight.data = weight - self.lr * self.vel[i]   #这步每个运算符都是重载后的
            weight.dirty = False
            
        if(self.t % self.decay_steps == 0):
            self.lr *= self.lr_decay



class Adam(optimizer):
    def __init__(self, weights: [pisor], lr, lr_decay=1, decay_steps=1000, beta1=0.9, beta2=0.999):
        super().__init__(weights, lr, lr_decay, decay_steps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.ms = [pisor.make_const(np.zeros(weight.numpy().shape), device=self.device) for weight in weights]
        self.vs = [pisor.make_const(np.zeros(weight.numpy().shape), device=self.device) for weight in weights] 
    
    def step(self):
        self.t += 1
        for i,weight in enumerate(self.weights):
            grad = weight.grad.detach()
            self.ms[i] = self.beta1*self.ms[i] + (1-self.beta1)*grad
            self.vs[i] = self.beta2*self.vs[i] + (1-self.beta2)*grad**2
            m_hat = self.ms[i] / (1-self.beta1**self.t)
            v_hat = self.vs[i] / (1-self.beta2**self.t)
            update_amount = -self.lr * m_hat / ((v_hat**0.5) + 1e-8)
            weight.data = weight + self.lr * update_amount
            weight.dirty = False
            
        if(self.t % self.decay_steps == 0):
            self.lr *= self.lr_decay   
