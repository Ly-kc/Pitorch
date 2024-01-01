#include "tensor_func.h"

Tensor relu_forward(const Tensor* input)
{
    int n = input->get_element_num();
    Tensor res = input->copy();
    if(input->get_device() == "cpu")
        relu_cpu(res.get_data(), input->read_data(), n);
    else
        relu_gpu(res.get_data(), input->read_data(), n);
    return res;
}

Tensor relu_backward(const Tensor* input, const Tensor* grad)
{
    int n = input->get_element_num();
    Tensor res = input->copy();
    
    if(input->get_device() == "cpu")
        backward_relu_cpu(res.get_data(), input->read_data(), n);
    else
        backward_relu_gpu(res.get_data(), input->read_data(), n);
    
    if(grad)
    {
        assert(input->get_device() == grad->get_device());
        if(grad->get_device() == "cpu")
            dot_cpu(res.get_data(), grad->read_data(), n);
        else
            dot_gpu(res.get_data(), grad->read_data(), n);
    }

    return res;
}

Tensor sigmoid_forward(const Tensor* input)
{
    int n = input->get_element_num();
    Tensor res = input->copy();

    if(input->get_device() == "cpu")
        sigmoid_cpu(res.get_data(), input->read_data(), n);
    else
        sigmoid_gpu(res.get_data(), input->read_data(), n);
    return res;
}

Tensor sigmoid_backward(const Tensor* forward_output, const Tensor* grad)
{    
    int n = forward_output->get_element_num();
    Tensor res = forward_output->copy();
    
    if(forward_output->get_device() == "cpu")
        backward_sigmoid_cpu(res.get_data(), forward_output->read_data(), n);
    else
        backward_sigmoid_gpu(res.get_data(), forward_output->read_data(), n);
    
    if(grad)
    {
        assert(forward_output->get_device() == grad->get_device());
        if(grad->get_device() == "cpu")
            dot_cpu(res.get_data(), grad->read_data(), n);
        else
            dot_gpu(res.get_data(), grad->read_data(), n);       
    }
    
    return res;
}

Tensor fc_forward(const Tensor* input, const Tensor* weight, const Tensor* bias)
{
    //input: batch_size * in_dim
    //weights: out_dim * in_dim
    //bias: out_dim
    //output: batch_size * out_dim

    assert(input->get_device() == weight->get_device());
    assert(input->get_device() == bias->get_device());
    int in_dim = input->get_shape()[1];
    int out_dim = weight->get_shape()[0];
    int batch_Size = input->get_shape()[0];
    assert(weight->get_shape()[1] == in_dim);
    assert(bias->get_shape()[0] == out_dim);

    std::string device = input->get_device();
    std::vector<int> out_shape = {batch_Size, out_dim}; 
    Tensor res(out_shape, device);
    if(device == "cpu")
    {
        fc_forward_cpu(input->read_data(),res.get_data(), weight->read_data(), bias->read_data(), in_dim, out_dim, batch_Size);
    }
    else
    {
        fc_forward_gpu(input->read_data(),res.get_data(), weight->read_data(), bias->read_data(), in_dim, out_dim, batch_Size);
    }
    return res;
}

void fc_backward(const Tensor* grad_y, const Tensor* input_x, const Tensor* weight, const Tensor* bias, Tensor* res_grad_x, Tensor* res_grad_weight, Tensor* res_grad_bias)
{
    //input: batch_size * in_dim
    //weights: out_dim * in_dim
    //bias: out_dim
    //grad_y: batch_size * out_dim
    //grad_x: batch_size * in_dim
    //grad_weight: out_dim * in_dim
    //grad_bias: out_dim
    assert(input_x->get_device() == weight->get_device());
    assert(input_x->get_device() == bias->get_device());
    assert(input_x->get_device() == grad_y->get_device());
    std::vector<int> y_shape = grad_y->get_shape();
    std::vector<int> x_shape = input_x->get_shape();
    std::vector<int> w_shape = weight->get_shape();
    std::vector<int> b_shape = bias->get_shape();
    int batch_Size = x_shape[0];
    int in_dim = x_shape[1];
    int out_dim = w_shape[0];
    assert(w_shape[1] == in_dim);
    assert(bias->get_shape()[0] == out_dim);
    assert(y_shape[0] == batch_Size);
    assert(y_shape[1] == out_dim);

    std::string device = input_x->get_device();
    Tensor grad_x(x_shape, device);
    // grad_x.print_information();
    Tensor grad_weight(w_shape, device);
    Tensor grad_bias(b_shape, device);

    if(device == "cpu")
    {
        fc_backward_cpu(grad_y->read_data(), input_x->read_data(), weight->read_data(), bias->read_data(), grad_x.get_data(), grad_weight.get_data(), grad_bias.get_data(), in_dim, out_dim, batch_Size);
    }
    else
    {
        fc_backward_gpu(grad_y->read_data(), input_x->read_data(), weight->read_data(), bias->read_data(), grad_x.get_data(), grad_weight.get_data(), grad_bias.get_data(), in_dim, out_dim, batch_Size);
    }
    res_grad_x->copy_from(grad_x);
    res_grad_weight->copy_from(grad_weight);
    res_grad_bias->copy_from(grad_bias);

}

Tensor conv2d_forward(const Tensor* input, const Tensor* weight, const Tensor* bias, int stride, int padding)
{
    assert(input->get_device() == weight->get_device());
    assert(input->get_device() == bias->get_device());
    int in_channel = input->get_shape()[1];
    int out_channel = weight->get_shape()[0];
    int kernel_size = weight->get_shape()[2];
    int batch_Size = input->get_shape()[0];
    int in_height = input->get_shape()[2];
    int in_width = input->get_shape()[3];
    assert((in_height + 2 * padding - kernel_size) % stride == 0);
    assert((in_width + 2 * padding - kernel_size) % stride == 0);
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    assert(weight->get_shape()[1] == in_channel);
    assert(bias->get_shape()[0] == out_channel);
    
    std::vector<int> out_shape = {batch_Size, out_channel, out_height, out_width};
    std::string device = input->get_device();
    Tensor res(out_shape, device);
    if(device == "cpu")
    {
        conv2d_forward_cpu(input->read_data(),res.get_data(), weight->read_data(), bias->read_data(),  
                            batch_Size, in_channel, out_channel, in_height, in_width, kernel_size, stride, padding);
    }
    else
    {
        conv2d_forward_gpu(input->read_data(),res.get_data(), weight->read_data(), bias->read_data(),  
                            batch_Size, in_channel, out_channel, in_height, in_width, kernel_size, stride, padding);
    }
    return res;
}


void conv2d_backward(const Tensor* grad_y, const Tensor* input_x, const Tensor* weight, int stride, int padding,
                    Tensor* grad_x, Tensor* grad_weight, Tensor* grad_bias){
    assert(input_x->get_device() == weight->get_device());
    assert(grad_y->get_device() == input_x->get_device());
    int in_channel = input_x->get_shape()[1];
    int out_channel = weight->get_shape()[0];
    int kernel_size = weight->get_shape()[2];
    int batch_Size = input_x->get_shape()[0];
    int in_height = input_x->get_shape()[2];
    int in_width = input_x->get_shape()[3];
    assert((in_height + 2 * padding - kernel_size) % stride == 0);
    assert((in_width + 2 * padding - kernel_size) % stride == 0);
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    assert(weight->get_shape()[1] == in_channel);
    assert(grad_y->get_shape()[0] == batch_Size);
    assert(grad_y->get_shape()[1] == out_channel);
    assert(grad_y->get_shape()[2] == out_height);
    assert(grad_y->get_shape()[3] == out_width);

    std::string device = input_x->get_device();
    std::vector<int> grad_x_shape = {batch_Size, in_channel, in_height, in_width};
    std::vector<int> grad_weight_shape = {out_channel, in_channel, kernel_size, kernel_size};
    std::vector<int> grad_bias_shape = {out_channel};
    Tensor grad_x_tensor(grad_x_shape, device);
    Tensor grad_weight_tensor(grad_weight_shape, device);
    Tensor grad_bias_tensor(grad_bias_shape, device);

    if(device == "cpu")
    {
        conv2d_backward_cpu(grad_y->read_data(), input_x->read_data(), weight->read_data(), 
                            grad_x_tensor.get_data(), grad_weight_tensor.get_data(), grad_bias_tensor.get_data(),
                            batch_Size, in_channel, out_channel, in_height, in_width, kernel_size, stride, padding);
    }
    else
    {
        conv2d_backward_gpu(grad_y->read_data(), input_x->read_data(), weight->read_data(), 
                            grad_x_tensor.get_data(), grad_weight_tensor.get_data(), grad_bias_tensor.get_data(),
                            batch_Size, in_channel, out_channel, in_height, in_width, kernel_size, stride, padding);
    }

    grad_x->copy_from(grad_x_tensor);
    grad_weight->copy_from(grad_weight_tensor);
    grad_bias->copy_from(grad_bias_tensor);
}

Tensor maxpool_forward(const Tensor* input, Tensor* mask, int K)
{
    int B = input->get_shape()[0];
    int C = input->get_shape()[1];
    int H = input->get_shape()[2];
    int W = input->get_shape()[3];
    assert(H%K == 0);
    assert(W%K == 0);

    std::vector<int> mask_shape = input->get_shape();
    std::vector<int> res_shape = {B, C, H/K, W/K};
    std::string device = input->get_device();
    Tensor res(res_shape, device);
    Tensor res_mask(mask_shape, device);

    if(device == "cpu")
    {
        maxpool_forward_cpu(input->read_data(), res.get_data(), res_mask.get_data(), B, C,  H, W, K);
    }
    else
    {
        maxpool_forward_gpu(input->read_data(), res.get_data(), res_mask.get_data(), B, C,  H, W, K);
    }
    mask->copy_from(res_mask);
    return res;
}   

void maxpool_backward(const Tensor* grad_y, const Tensor* mask, Tensor* grad_x)
{
    int B = grad_y->get_shape()[0];
    int C = grad_y->get_shape()[1];
    int newH = grad_y->get_shape()[2];
    int newW = grad_y->get_shape()[3];
    int H = mask->get_shape()[2];
    int W = mask->get_shape()[3];
    assert(H%newH == 0);
    assert(W%newW == 0);
    int K = H/newH;
    assert(K == W/newW);
    assert(B == mask->get_shape()[0]);
    assert(C == mask->get_shape()[1]);
    std::vector<int> res_shape = {B,C,H,W};
    std::string device = grad_y->get_device();
    Tensor res(res_shape, device);
    if(device == "cpu")
    {
        maxpool_backward_cpu(grad_y->read_data(), mask->read_data(), res.get_data(), B, C , H, W, K);
    }
    else
    {
        maxpool_backward_gpu(grad_y->read_data(), mask->read_data(), res.get_data(), B, C , H, W, K);
    }
    grad_x->copy_from(res);
}

Tensor softmax_forward(const Tensor* input)
{
    int N = input->get_shape()[0];
    int C = input->get_shape()[1];
    std::string device = input->get_device();
    std::vector<int> out_shape = {N,C};
    Tensor res(out_shape,device);

    if(device == "cpu")
    {
        softmax_forward_cpu(input->read_data(), res.get_data(), N, C);
    }
    else
    {
        softmax_forward_gpu(input->read_data(), res.get_data(), N, C);
    }
    return res;
}

Tensor cross_entropy_forward(const Tensor* input, const Tensor* gt_prob)
{
    //output: (1,)
    int N = input->get_shape()[0];
    int C = input->get_shape()[1];
    assert(gt_prob->get_shape()[0] == N);
    assert(gt_prob->get_shape()[1] == C);
    std::string device = input->get_device();
    std::vector<int> out_shape = {1};
    Tensor res(out_shape,device);

    if(device == "cpu")
    {
        cross_entropy_forward_cpu(input->read_data(), gt_prob->read_data(), res.get_data(), N, C);
    }
    else
    {
        cross_entropy_forward_gpu(input->read_data(), gt_prob->read_data(), res.get_data(), N, C);
    }
    return res;

}

void cross_entropy_with_softmax_backward(const Tensor* prob, const Tensor* gt_prob, Tensor* grad)
{
    //grad: (N,C)
    int N = prob->get_shape()[0];
    int C = prob->get_shape()[1];
    assert(gt_prob->get_shape()[0] == N);
    assert(gt_prob->get_shape()[1] == C);
    std::vector<int> out_shape = {N,C};
    std::string device = prob->get_device();
    Tensor res(out_shape,device);
    if(device == "cpu")
    {
        cross_entropy_with_softmax_backward_cpu(prob->read_data(), gt_prob->read_data(), res.get_data(), N, C);
    }
    else
    {
        cross_entropy_with_softmax_backward_gpu(prob->read_data(), gt_prob->read_data(), res.get_data(), N, C);
    }
    grad->copy_from(res);
}