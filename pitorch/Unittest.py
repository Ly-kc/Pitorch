import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.functional import F
import unittest
from tqdm import tqdm
import mytensor
from mytensor import raw_pisor as Tensor

class net_test(unittest.TestCase):
    def setUp(self):
        self.device = 'gpu'
    
    def test_fc_forward(self):
        for i in tqdm(range(20000),desc='fc'):
            #rand shape and value
            batch_size = np.random.randint(1, 10)
            in_dim = np.random.randint(1, 500)
            out_dim = np.random.randint(1, 500)
            input = np.random.rand(batch_size, in_dim)
            weights = np.random.rand(out_dim, in_dim)
            bias = np.random.rand(out_dim)
            # print(batch_size,in_dim,out_dim)
            scalar = np.random.randint(1,100)
            input *= scalar
            #output from torch
            input_pytorch = torch.tensor(input, device='cuda', requires_grad=True)
            weights_pytorch = torch.tensor(weights, device='cuda', requires_grad=True)
            bias_pytorch = torch.tensor(bias, device='cuda', requires_grad=True)    
            output_pytorch = F.linear(input_pytorch, weights_pytorch, bias_pytorch)
            np_output_pytorch = output_pytorch.detach().cpu().numpy()
            #output frmo ours
            ''' 
            our forward:
            input: batch_size * in_dim
            weights: out_dim * in_dim
            bias: out_dim
            output: batch_size * out_dim
            '''
            input_tensor = Tensor(input, 'cpu')
            weights_tensor = Tensor(weights, 'cpu')
            bias_tensor = Tensor(bias, 'cpu')
            # print(input_tensor.shape(), weights_tensor.shape(), bias_tensor.shape())
            output_mytensor = mytensor.fc_forward(input_tensor, weights_tensor, bias_tensor)
            np_output_mytensor = output_mytensor.numpy()
            
            assert np.allclose(np_output_pytorch, np_output_mytensor, rtol=1e-3, atol=1e-3)
            
            #back propagation
            
            grad_y = np.random.rand(batch_size, out_dim)    
            #grad from torch
            grad_y_pytorch = torch.tensor(grad_y, device='cuda')
            loss = torch.sum(output_pytorch * grad_y_pytorch)
            loss.backward()
            grad_x_pytorch = input_pytorch.grad.detach().cpu().numpy()
            grad_weight_pytorch = weights_pytorch.grad.detach().cpu().numpy()
            grad_bias_pytorch = bias_pytorch.grad.detach().cpu().numpy()

            #grad from mytensor
            '''
            //grad_y: batch_size * out_dim
            //grad_x: batch_size * in_dim
            //grad_weight: out_dim * in_dim
            //grad_bias: out_dim
            '''
            grad_y_tensor = Tensor(grad_y, 'cpu')
            grad_x_mytensor, grad_weight_mytensor, grad_mybias_tensor = mytensor.fc_backward(grad_y_tensor, input_tensor, weights_tensor, bias_tensor)
            grad_x_mytensor, grad_weight_mytensor, grad_mybias_tensor = grad_x_mytensor.numpy(), grad_weight_mytensor.numpy(), grad_mybias_tensor.numpy()
            
            assert np.allclose(grad_x_pytorch, grad_x_mytensor, rtol=1e-3, atol=1e-3)
            assert np.allclose(grad_weight_pytorch, grad_weight_mytensor, rtol=1e-3, atol=1e-3)
            assert np.allclose(grad_bias_pytorch, grad_mybias_tensor, rtol=1e-3, atol=1e-3)
    
    def test_conv(self):
        '''
        our forward:
        //input: B * C_in * H * W
        //weights: C_out * C_in * K * K
        //bias: C_out
        //output: B * C_out * H_out * W_out
        # assert((in_height + 2 * padding - kernel_size) % stride == 0);
        
        our backward:
        //grad_y: B * C_out * H_out * W_out
        //input_x, grad_x: B * C_in * H * W
        //weights, grad_weights: C_out * C_in * K * K
        //grad_bias: C_out
        '''
        for i in tqdm(range(20000),desc='conv'):
            batch_size = np.random.randint(1, 10)
            in_channel = np.random.randint(1, 100)
            out_channel = np.random.randint(1, 100)
            kernel_size = np.random.randint(1, 10)
            stride = np.random.randint(1, kernel_size + 1)
            padding = np.random.randint(1, kernel_size + 1)
            min_snum = max(0,int((2*padding - kernel_size) // stride + 1))
            hs = np.random.randint(min_snum, min_snum + 4)
            ws = np.random.randint(min_snum, min_snum + 4)
            height = stride * hs + kernel_size - 2* padding
            width = stride * ws + kernel_size - 2* padding
            # batch_size,in_channel,out_channel,kernel_size,stride,padding,height,width = 10,100,100,5,8,6,17,1
            input = np.random.rand(batch_size, in_channel, height, width)
            weights = np.random.rand(out_channel, in_channel, kernel_size, kernel_size)
            bias = np.random.rand(out_channel)
            scalar = np.random.randint(1,10)
            input *= scalar
            weights *= scalar
            bias *= scalar
            # print(i)
            # print(batch_size,in_channel,out_channel,kernel_size,stride,padding,height,width)
            # print('forward')
            #our forward
            input_mytensor = Tensor(input, self.device)
            weights_mytensor = Tensor(weights, self.device)
            # bias_mytensor = Tensor(bias)
            bias_mytensor = Tensor(bias, self.device)
            
            output_mytensor = mytensor.Convolution_forward(input_mytensor,weights_mytensor,bias_mytensor, stride, padding)

            #pytorch
            input_pytorch = torch.tensor(input, device='cuda', requires_grad=True)
            weights_pytorch = torch.tensor(weights, device='cuda', requires_grad=True)
            bias_pytorch = torch.tensor(bias, device='cuda', requires_grad=True)
            output_pytorch = F.conv2d(input_pytorch, weights_pytorch, bias_pytorch, stride, padding)
            
            if(not np.allclose(output_mytensor.numpy(), output_pytorch.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)):
                print("batch_size: ", batch_size)
                print("in_channel: ", in_channel)
                print("out_channel: ", out_channel)
                print("kernel_size: ", kernel_size)
                print("stride: ", stride)
                print("padding: ", padding)
                print("height: ", height)
                print("width: ", width)
                print("input: ", input)
                print("weights: ", weights)
                print("bias: ", bias)
                print("output_mytensor: ", output_mytensor.numpy())
                print("output_pytorch: ", output_pytorch.detach().cpu().numpy())
                assert False

            #backward
            # print('backward')
            grad_y = np.random.rand(*output_pytorch.shape)
            #pytorch
            loss = torch.sum(output_pytorch * torch.tensor(grad_y, device='cuda'))
            loss.backward()
            grad_x_pytorch = input_pytorch.grad
            grad_weight_pytorch = weights_pytorch.grad
            grad_bias_pytorch = bias_pytorch.grad
            #ours
            grad_y_tensor = Tensor(grad_y, self.device)
            grad_x_mytensor, grad_weight_mytensor, grad_mybias_tensor = mytensor.Convolution_backward(grad_y_tensor, input_mytensor, weights_mytensor, stride, padding)
            
            assert np.allclose(grad_x_pytorch.detach().cpu().numpy(), grad_x_mytensor.numpy(), rtol=1e-3, atol=1e-3)
            assert np.allclose(grad_weight_pytorch.detach().cpu().numpy(), grad_weight_mytensor.numpy(), rtol=1e-3, atol=1e-3)
            assert np.allclose(grad_bias_pytorch.detach().cpu().numpy(), grad_mybias_tensor.numpy(), rtol=1e-3, atol=1e-3)

    def test_pooling(self):
        for i in tqdm(range(5000),desc='pooling'):
            batch_size = np.random.randint(1, 10)
            in_channel = np.random.randint(1, 100)
            kernel_size = np.random.randint(1, 10)
            hs = np.random.randint(1, 50)
            ws = np.random.randint(1, 50)
            # batch_size,in_channel,hs,ws,kernel_size = 1,10,40,50,2
            height = kernel_size * hs
            width = kernel_size * ws
            input = np.random.rand(batch_size, in_channel, height, width)
            scalar = np.random.randint(1,100)
            input *= scalar
            #forward
            #ours
            input_mytensor = Tensor(input, self.device)
            output_mytensor, mask_mytensor = mytensor.Pooling_forward(input_mytensor, kernel_size)

            #pytorch
            input_pytorch = torch.tensor(input, device='cuda', requires_grad=True, dtype=torch.float32)
            output_pytorch, mask_pytorch = F.max_pool2d(input_pytorch, kernel_size, return_indices=True)
            
            # print(batch_size, in_channel,kernel_size, height, width, mask_pytorch.shape)
            if (not np.allclose(output_mytensor.numpy(), output_pytorch.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)):
                print("batch_size: ", batch_size)
                print("in_channel: ", in_channel)
                print("kernel_size: ", kernel_size)
                print("height: ", height)
                print("width: ", width)
                print("input: ", input)
                print("output_mytensor: ", output_mytensor.numpy())
                print("output_pytorch: ", output_pytorch.detach().cpu().numpy())
                print("mask_mytensor: ", mask_mytensor.numpy())
                print("mask_pytorch: ", mask_pytorch.detach().cpu().numpy())
                assert False
            
            #mask
            #以下转换在channel>0的时候似乎有误
            #transfer pytorch mask to binary mask
            mask_pytorch1 = mask_pytorch.detach().cpu().numpy()
            mask_binary= np.zeros_like(input).reshape(-1)
            mask_binary[mask_pytorch1.reshape(-1)] = 1
            
            #注意float精度会影响mask结果
            # if (not np.allclose(mask_mytensor.numpy().reshape(-1), mask_binary, rtol=1e-3, atol=1e-3)):
            #     print("batch_size: ", batch_size)
            #     print("in_channel: ", in_channel)
            #     print("kernel_size: ", kernel_size)
            #     print("height: ", height)
            #     print("width: ", width)
            #     print("input: ", input)
            #     dis = mask_mytensor.numpy() - mask_binary.reshape(*mask_mytensor.shape())
            #     place = (dis > 1e-3).nonzero()
            #     print(place)
            #     h,w = place[2][0]//kernel_size, place[3][0]//kernel_size
            #     #输出该处的值
            #     print("partial input: ", input[0,place[1][0],h*kernel_size:(h+1)*kernel_size,w*kernel_size:(w+1)*kernel_size])
            #     print("partial pytorch mask: ", mask_binary.reshape(*mask_mytensor.shape())[0,place[1][0],h*kernel_size:(h+1)*kernel_size,w*kernel_size:(w+1)*kernel_size])
            #     print("partial our mask: ", mask_mytensor.numpy()[0,place[1][0],h*kernel_size:(h+1)*kernel_size,w*kernel_size:(w+1)*kernel_size])
                
            #     assert False 
            
            #backward
            grad_y = np.random.rand(*output_pytorch.shape)   
            #pytorch
            loss = torch.sum(output_pytorch * torch.tensor(grad_y, device='cuda'))
            loss.backward()
            grad_x_pytorch = input_pytorch.grad
            #ours
            grad_y_tensor = Tensor(grad_y, self.device)
            grad_x_mytensor = mytensor.Pooling_backward(grad_y_tensor, mask_mytensor)
            
            if not np.allclose(grad_x_pytorch.detach().cpu().numpy(), grad_x_mytensor.numpy(), rtol=1e-3, atol=1e-3):
                print("batch_size: ", batch_size)
                print("in_channel: ", in_channel)
                print("kernel_size: ", kernel_size)
                print("height: ", height)
                print("width: ", width)
                print("input: ", input)
                print("grad_y: ",grad_y)
                print("grad_x_mytensor: ", grad_x_mytensor.numpy())
                print("grad_x_pytorch: ", grad_x_pytorch.detach().cpu().numpy())
                dis = grad_x_pytorch.detach().cpu().numpy() - grad_x_mytensor.numpy()
                print("dis: " ,dis[dis > 1e-3])
                print((dis>1e-3).nonzero())
                print(grad_x_pytorch.detach().cpu().numpy()[dis > 1e-3])
                print(grad_x_mytensor.numpy()[dis > 1e-3])
                assert False
            
    def test_softmax_crossentropy(self):
        for i in tqdm(range(20000),desc='softmax_crossentropy'):
            batch_size = np.random.randint(1, 10)
            in_dim = np.random.randint(1, 100)
            batch_size,in_dim = 1,2
            input = np.random.rand(batch_size, in_dim)
            scalar = np.random.randint(1,100)
            input *= scalar
            #softmax forward
            #ours
            input_mytensor = Tensor(input, self.device)
            output_mytensor = mytensor.Softmax_forward(input_mytensor)
            #pytorch
            input_pytorch = torch.tensor(input, device='cuda', requires_grad=True)
            output_pytorch = F.softmax(input_pytorch, dim=1)
            assert np.allclose(output_mytensor.numpy(), output_pytorch.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        
            #crossentropy forward
            prob_gt = np.random.rand(batch_size, in_dim)
            prob_gt = prob_gt / np.sum(prob_gt, axis=1, keepdims=True)
            #ours
            prob_gt_mytensor = Tensor(prob_gt, self.device)
            loss_mytensor = mytensor.Crossentropy_forward(output_mytensor, prob_gt_mytensor)
            #pytorch
            prob_gt_pytorch = torch.tensor(prob_gt, device='cuda')
            loss_pytorch = F.cross_entropy(input_pytorch, prob_gt_pytorch)   #pytorch's crossentropy has softmax inside
            
            if not np.allclose(loss_mytensor.numpy(), loss_pytorch.detach().cpu().numpy(), rtol=1e-3, atol=1e-3):
                print("batch_size: ", batch_size)
                print("in_dim: ", in_dim)
                # print("input: ", input)
                print("output: ",output_mytensor.numpy())
                print("prob_gt: ", prob_gt)
                print("loss_mytensor: ", loss_mytensor.numpy())
                print("loss_pytorch: ", loss_pytorch.detach().cpu().numpy())
                assert False
        
            #crossentropy backward with softmax backward
            #ours
            grad_x_mytensor = mytensor.Softmax_Crossentropy_backward(output_mytensor, prob_gt_mytensor)
            #pytorch
            loss_pytorch.backward()
            grad_x_pytorch = input_pytorch.grad
            
            assert np.allclose(grad_x_mytensor.numpy(), grad_x_pytorch.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
            

if __name__ == "__main__":
    unittest.main()