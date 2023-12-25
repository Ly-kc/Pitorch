#include "Tensor.h"
#include "tensor_func.h"
#include "utils.h"

void test_initialize_tensor()
{
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "---------------test initialize tensor: ---------------" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    //check if the shape is initialized correctly
    std::vector<int> shape{2,3, 4};
    std::cout << "shape: " << std::endl;
    for(int i = 0 ; i < shape.size() ; i++)
    {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;

    //initialized tensor through the shape only
    Tensor tensor1(shape);   
    std::cout << "--------------tensor1: --------------" << std::endl;
    tensor1.print_information();
    std::cout << std::endl;

    //initialized tensor through the shape and the data
    float* data = new float[24];
    for(int i = 0 ; i < 24 ; i++)
    {
        data[i] = float(i);
    }
    Tensor tensor2(shape, data);
    std::cout << "--------------tensor2: --------------" << std::endl;
    tensor2.print_information();
    std::cout << std::endl;

    //initialized tensor through the shape and scalar
    Tensor tensor3(shape, 1.0);
    std::cout << "--------------tensor3: --------------" << std::endl;
    tensor3.print_information();
    std::cout << std::endl;

    //initialized tensor on gpu through shape
    Tensor tensor4(shape, "gpu");
    std::cout << "--------------tensor4: --------------" << std::endl;
    tensor4.print_information();
    std::cout << std::endl;

    //initialized tensor on gpu through shape and data
    Tensor tensor5(shape, data, "gpu");
    std::cout << "--------------tensor5: --------------" << std::endl;
    tensor5.print_information();
    std::cout << std::endl;

    //initialized tensor on gpu through shape and scalar
    Tensor tensor6(shape, 1.0, "gpu");
    std::cout << "--------------tensor6: --------------" << std::endl;
    tensor6.print_information();
    std::cout << std::endl;

    //move tensor from cpu to gpu
    tensor2._gpu();
    std::cout << "--------------transfer tensor2 to gpu: --------------" << std::endl;
    tensor2.print_information();
    std::cout << std::endl;

    //move tensor from gpu to cpu
    tensor2._cpu();
    std::cout << "--------------transfer tensor2 back to cpu: --------------" << std::endl;
    tensor2.print_information();
    std::cout << std::endl;

    //copy constructor
    Tensor g = tensor2.gpu();
    std::cout << "--------------transfer tensor2 to gpu and assign it to a new tensor: --------------" << std::endl;
    g.print_information();
    std::cout << std::endl;
    Tensor c = tensor2.cpu();
    std::cout << "--------------transfer tensor2 back to cpu and assign it to a new tensor: --------------" << std::endl;
    c.print_information();
    std::cout << std::endl;
}

void test_activate_func()
{
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "---------------test activate function: ---------------" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::vector<int> shape{2, 3};
    float* data = new float[6];
    for(int i = 0 ; i < 6 ; i++)
    {
        data[i] = float(i);
    }
    Tensor tensor2(shape, data);

    //relu forward on cpu
    Tensor relu_input(tensor2-3);
    Tensor relu_output = relu_forward(&relu_input);
    std::cout << "--------------relu_input_cpu: --------------" << std::endl;
    relu_input.print_information();
    std::cout << "--------------relu_output_cpu: --------------" << std::endl;
    relu_output.print_information();
    std::cout << std::endl;

    //relu backward on cpu
    Tensor last_grad(tensor2+10);
    Tensor relu_grad = relu_backward(&relu_input,&last_grad);
    std::cout << "--------------grad from last layer: --------------" << std::endl;
    last_grad.print_information();
    std::cout << "--------------relu_grad_cpu: --------------" << std::endl;
    relu_grad.print_information();
    std::cout << std::endl;

    //relu forward on gpu
    Tensor relu_input_gpu = relu_input.gpu();
    Tensor relu_output_gpu = relu_forward(&relu_input_gpu);
    std::cout << "--------------relu_input_gpu: --------------" << std::endl;
    relu_input_gpu.print_information();
    std::cout << "--------------relu_output_gpu: --------------" << std::endl;
    relu_output_gpu.print_information();
    std::cout << std::endl;
    
    //relu backward on gpu
    Tensor last_grad_gpu = last_grad.gpu();
    Tensor relu_grad_gpu = relu_backward(&relu_input_gpu,&last_grad_gpu);
    std::cout << "--------------grad from last layer: --------------" << std::endl;
    last_grad_gpu.print_information();
    std::cout << "--------------relu_grad_gpu: --------------" << std::endl;
    relu_grad_gpu.print_information();
    std::cout << std::endl;

    //sigmoid forward on cpu
    Tensor sigmoid_input(tensor2/10);
    Tensor sigmoid_output = sigmoid_forward(&sigmoid_input);
    std::cout << "--------------sigmoid_input: --------------" << std::endl;
    sigmoid_input.print_information();
    std::cout << "--------------sigmoid_output: --------------" << std::endl;
    sigmoid_output.print_information();
    std::cout << std::endl;

    //sigmoid backward on cpu
    Tensor sigmoid_grad = sigmoid_backward(&sigmoid_output,&last_grad);
    std::cout << "--------------grad from last layer: --------------" << std::endl;
    last_grad.print_information();
    std::cout << "--------------sigmoid_grad: --------------" << std::endl;
    sigmoid_grad.print_information();
    std::cout << std::endl;
    
    //sigmoid forward on gpu
    Tensor sigmoid_input_gpu = sigmoid_input.gpu();
    Tensor sigmoid_output_gpu = sigmoid_forward(&sigmoid_input_gpu);
    std::cout << "--------------sigmoid_input_gpu: --------------" << std::endl;
    sigmoid_input_gpu.print_information();
    std::cout << "--------------sigmoid_output_gpu: --------------" << std::endl;
    sigmoid_output_gpu.print_information();
    std::cout << std::endl;

    //sigmoid backward on gpu
    Tensor sigmoid_grad_gpu = sigmoid_backward(&sigmoid_output_gpu,&last_grad_gpu);
    std::cout << "--------------grad from last layer: --------------" << std::endl;
    last_grad_gpu.print_information();
    std::cout << "--------------sigmoid_grad_gpu: --------------" << std::endl;
    sigmoid_grad_gpu.print_information();
    std::cout << std::endl;
}

void test_data_manager()
{
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "---------------test data manager: ---------------" << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    //initialized tensor through the shape and scalar
    std::vector<int> shape{2, 3};
    Tensor tensor1(shape, 1.0);
    std::cout << "--------------tensor1: --------------" << std::endl;
    tensor1.print_information();
    std::cout << std::endl;

    //copy constructor
    Tensor tensor2(tensor1);
    std::cout << "--------------tensor2: --------------" << std::endl;
    tensor2.print_information();
    std::cout << std::endl;

    //copy assignment
    Tensor tensor3 = tensor1;
    std::cout << "--------------tensor3: --------------" << std::endl;
    tensor3.print_information();
    std::cout << std::endl;

    //pointer
    Tensor* tensor4 = new Tensor(tensor2);
    std::cout << "--------------tensor4: --------------" << std::endl;
    tensor4->print_information();
    std::cout << std::endl;

    delete tensor4;
    
    tensor3 = tensor2;
    std::cout << "--------------tensor3: --------------" << std::endl;
    tensor3.print_information();
    std::cout << std::endl;

}

void test_gemm()
{
    std::cout  << "------------------------------------------------------" << std::endl;
    std::cout  << "---------------test gemm: ---------------" << std::endl;
    std::cout  << "------------------------------------------------------" << std::endl;
    //generate two matrix
    int M = 2;
    int K = 4;
    int N = 3;
    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C = new float[M*N];

    for(int i = 0 ; i < M*K ; i++)
    {
        A[i] = float(i);
    }
    for(int i = 0 ; i < K*N ; i++)
    {
        B[i] = float(i);
    }
    fill_cpu(C,1.0,M*N);

    std::cout << "--------------A: --------------" << std::endl;
    for(int i = 0 ; i < M ; i ++)
    {
        for(int j = 0 ; j < K ; j++)
        {
            std::cout << A[i*K+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------B: --------------" << std::endl;
    for(int i = 0 ; i < K ; i ++)
    {
        for(int j = 0 ; j < N ; j++)
        {
            std::cout << B[i*N+j] << " ";
        }
        std::cout << std::endl;
    }

    gemm_cpu(false,false,A, B, C, M, N, K, 1.0, 10.0);
    std::cout << "--------------C: --------------" << std::endl;
    for(int i = 0 ; i < M ; i ++)
    {
        for(int j = 0 ; j < N ; j++)
        {
            std::cout << C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * K * N);    
    cudaMalloc(&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    fill_gpu(d_C,2.0,M*N);
    gemm_gpu(false,false,d_A, d_B, d_C, M, N, K, 1.0, 10.0);
    cudaMemcpy(C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    std::cout << "--------------C: --------------" << std::endl;
    for(int i = 0 ; i < M ; i ++)
    {
        for(int j = 0 ; j < N ; j++)
        {
            std::cout << C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;

    //test transpose
    std::cout << "--------------test transpose: --------------" << std::endl;
    M = 2;
    K = 3;
    N = 2;
    A = new float[M*K];
    B = new float[K*N];
    C = new float[100];

    for(int i = 0 ; i < M*K ; i++)
    {
        A[i] = float(i);
    }
    for(int i = 0 ; i < K*N ; i++)
    {
        B[i] = float(i);
    }
    fill_cpu(C,1.0,100);

    std::cout << "--------------A before transpose: --------------" << std::endl;
    for(int i = 0 ; i < M ; i ++)
    {
        for(int j = 0 ; j < K ; j++)
        {
            std::cout << A[i*K+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------A after transpose: --------------" << std::endl;
    for(int i = 0 ; i < K ; i ++)
    {
        for(int j = 0 ; j < M ; j++)
        {
            std::cout << A[j*K+i] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "--------------B before transpose: --------------" << std::endl;
    for(int i = 0 ; i < K ; i ++)
    {
        for(int j = 0 ; j < N ; j++)
        {
            std::cout << B[i*N+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------B after transpose: --------------" << std::endl;
    for(int i = 0 ; i < N ; i ++)
    {
        for(int j = 0 ; j < K ; j++)
        {
            std::cout << B[j*N+i] << " ";
        }
        std::cout << std::endl;
    }

    gemm_cpu(false,false,A, B, C, M, N, K, 1.0, 0.0);
    std::cout << "--------------C = A @ B: --------------" << std::endl;
    for(int i = 0 ; i < M ; i ++)
    {
        for(int j = 0 ; j < N ; j++)
        {
            std::cout << C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }
    gemm_cpu(true,true,A, B, C, K, K, M, 1.0, 0.0);
    std::cout << "--------------C = A.T @ B.T: --------------" << std::endl;
    for(int i = 0 ; i < K ; i ++)
    {
        for(int j = 0 ; j < K ; j++)
        {
            std::cout << C[i*K+j] << " ";
        }
        std::cout << std::endl;
    }

    M = 2;
    N = 4;
    K = 3;
    A = new float[M*K]; //A (K,M)
    B = new float[K*N]; //(K,N)
    C = new float[M*N];
    for(int i = 0 ; i < M*K ; i++)
    {
        A[i] = float(i);
    }
    for(int i = 0 ; i < K*N ; i++)
    {
        B[i] = float(i);
    }
    std::cout << "--------------A: --------------" << std::endl;
    for(int i = 0 ; i < K ; i ++)
    {
        for(int j = 0 ; j < M ; j++)
        {
            std::cout << A[i*M + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "--------------B: --------------" << std::endl;
    for(int i = 0 ; i < K ; i ++)
    {
        for(int j = 0 ; j < N ; j++)
        {
            std::cout << B[i*N+j] << " ";
        }
        std::cout << std::endl;
    }

    d_A = new float[M*K];
    d_B = new float[K*N];
    d_C = new float[M*N];
    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * K * N);    
    cudaMalloc(&d_C, sizeof(float) * M*N);

    cudaMemcpy(d_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    fill_gpu(d_C,2.0,M*N);
    
    gemm_gpu(true,false,d_A, d_B, d_C, M, N, K, 1.0, 0.0);
    cudaMemcpy(C, d_C, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
    std::cout << "--------------C = A.T @ B on gpu: --------------" << std::endl;
    for(int i = 0 ; i < M ; i ++)
    {
        for(int j = 0 ; j < N ; j++)
        {
            std::cout << C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }

    gemm_gpu(true,false,d_B,d_A, d_C, N, M, K, 1.0, 0.0);
    cudaMemcpy(C, d_C, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
    std::cout << "--------------C = B.T @ A on gpu: --------------" << std::endl;
    for(int i = 0 ; i < N ; i ++)
    {
        for(int j = 0 ; j < M ; j++)
        {
            std::cout << C[i*M+j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void test_fc()
{
    std::cout  << "------------------------------------------------------" << std::endl;
    std::cout  << "---------------test fc: ---------------" << std::endl;
    std::cout  << "------------------------------------------------------" << std::endl;
    int x_feature_dim = 3;
    int batch_size = 2;
    int y_feature_dim = 4;
    std::vector<int> x_shape = {batch_size, x_feature_dim};
    std::vector<int> weights_shape = {y_feature_dim, x_feature_dim};
    std::vector<int> bias_shape = {y_feature_dim};
    Tensor x(x_shape);
    Tensor weight(weights_shape);
    Tensor bias(bias_shape);
    for(int i = 0 ; i < x.get_element_num() ; i ++)
        x.get_data()[i] = float(i);
    for(int i = 0 ; i < weight.get_element_num() ; i ++)
        weight.get_data()[i] = float(i)*2;
    for(int i = 0 ; i < bias.get_element_num() ; i ++)
        bias.get_data()[i] = float(i);
        // bias.get_data()[i] = 0.0;
    Tensor y = fc_forward(&x, &weight, &bias);
    std::cout << "--------------x on cpu: --------------" << std::endl;
    x.print_data();
    std::cout << "--------------weight on cpu: --------------" << std::endl;
    weight.print_data();
    std::cout << "--------------bias on cpu: --------------" << std::endl;
    bias.print_data();
    std::cout << "--------------y on cpu: --------------" << std::endl;
    y.print_data();
    std::cout << std::endl;

    Tensor y_grad = y.copy();
    for(int i = 0 ; i < y_grad.get_element_num() ; i ++)
        y_grad.get_data()[i] = i%3;
    std::cout << "--------------y_grad on cpu: --------------" << std::endl;
    y_grad.print_data();

    Tensor x_grad,w_grad,b_grad;
    fc_backward(&y_grad, &x, &weight, &bias, &x_grad, &w_grad, &b_grad);
    std::cout << "--------------x_grad on cpu: --------------" << std::endl;
    x_grad.print_data();
    std::cout << "--------------w_grad on cpu: --------------" << std::endl;
    w_grad.print_data();
    std::cout << "--------------b_grad on cpu: --------------" << std::endl;
    b_grad.print_data();
    std::cout << std::endl;


    x._gpu();
    weight._gpu();
    bias._gpu();
    y_grad._gpu();
    Tensor y_gpu = fc_forward(&x, &weight, &bias);
    std::cout << "--------------x on gpu: --------------" << std::endl;
    x.print_data();
    std::cout << "--------------weight on gpu: --------------" << std::endl;
    weight.print_data();
    std::cout << "--------------bias on gpu: --------------" << std::endl;
    bias.print_data();
    std::cout << "--------------y on gpu: --------------" << std::endl;
    y_gpu.print_data();
    std::cout << std::endl;

    Tensor x_grad_gpu,w_grad_gpu,b_grad_gpu;
    fc_backward(&y_grad, &x, &weight, &bias, &x_grad_gpu, &w_grad_gpu, &b_grad_gpu);
    std::cout << "--------------y_grad on gpu: --------------" << std::endl;
    y_grad.print_data();
    std::cout << "--------------x_grad on gpu: --------------" << std::endl;
    x_grad_gpu.print_data();
    std::cout << "--------------w_grad on gpu: --------------" << std::endl;
    w_grad_gpu.print_data();
    std::cout << "--------------b_grad on gpu: --------------" << std::endl;
    b_grad_gpu.print_data();
    std::cout << std::endl;
    sync_and_check_cuda_error_force();
}

void test_conv()
{
    int x_channel = 2;
    int x_height = 3;
    int x_width = 5;
    int batch_size = 2;
    int y_channel = 2;
    int stride = 2;
    int padding = 1;
    int kernel_size = 3;

    kernel_size=5;
    stride = 8;
    padding = 6;
    x_height = 17;
    x_width = 1;
    std::vector<int> x_shape = {batch_size, x_channel, x_height, x_width};
    std::vector<int> weights_shape = {y_channel, x_channel, kernel_size, kernel_size};
    std::vector<int> bias_shape = {y_channel};
    Tensor x(x_shape);
    Tensor weight(weights_shape);
    Tensor bias(bias_shape);
    for(int i = 0 ; i < x.get_element_num() ; i ++)
        x.get_data()[i] = float(i);
    for(int i = 0 ; i < weight.get_element_num() ; i ++)
        weight.get_data()[i] = 1;
    for(int i = 0 ; i < bias.get_element_num() ; i ++)
        bias.get_data()[i] = 10.0;
    
    Tensor y = conv2d_forward(&x, &weight, &bias, stride, padding);
    std::cout << "--------------x on cpu: --------------" << std::endl;
    x.print_data();
    std::cout << "--------------weight on cpu: --------------" << std::endl;
    weight.print_data();
    std::cout << "--------------bias on cpu: --------------" << std::endl;
    bias.print_data();

    Tensor y_grad = y.copy();
    for(int i = 0 ; i < y_grad.get_element_num() ; i ++)
        y_grad.get_data()[i] = i%3;
    std::cout << "--------------y_grad on cpu: --------------" << std::endl;
    y_grad.print_data();
    std::cout << std::endl;

    std::cout << "--------------y on cpu: --------------" << std::endl;
    y.print_data();
    Tensor x_grad,w_grad,b_grad;
    conv2d_backward(&y_grad, &x, &weight, stride, padding, &x_grad, &w_grad, &b_grad);
    std::cout << "--------------x_grad on cpu: --------------" << std::endl;
    x_grad.print_data();
    std::cout << "--------------w_grad on cpu: --------------" << std::endl;
    w_grad.print_data();
    std::cout << "--------------b_grad on cpu: --------------" << std::endl;
    b_grad.print_data();
    std::cout << std::endl;

    x._gpu();
    weight._gpu();
    bias._gpu();
    y_grad._gpu();
    Tensor y_gpu = conv2d_forward(&x, &weight, &bias, stride, padding);
    std::cout << "--------------y on gpu: --------------" << std::endl;
    y_gpu.print_data();
    conv2d_backward(&y_grad, &x, &weight, stride, padding, &x_grad, &w_grad, &b_grad);
    std::cout << "--------------x_grad on gpu: --------------" << std::endl;
    x_grad.print_data();
    std::cout << "--------------w_grad on gpu: --------------" << std::endl;
    w_grad.print_data();
    std::cout << "--------------b_grad on gpu: --------------" << std::endl;
    b_grad.print_data();
    std::cout << std::endl;
}

void test_pooling()
{
    int K = 2;
    int x_channel = 2;
    int x_height = 2;
    int x_width = 4;
    int batch_size = 2;

    std::vector<int> x_shape = {batch_size, x_channel, x_height, x_width};  
    Tensor x(x_shape);
    for(int i = 0 ; i < x.get_element_num() ; i ++)
        x.get_data()[i] = (i-5)*(i-5);
    
    std::cout << "--------------x on cpu: --------------" << std::endl;
    x.print_data();

    Tensor mask;
    Tensor y = maxpool_forward(&x, &mask, K);
    Tensor grad_y = y.copy();
    for(int i = 0 ; i < y.get_element_num() ; i ++)
        grad_y.get_data()[i] = 10-i;
    std::cout << "--------------y grad on cpu: --------------" << std::endl;
    grad_y.print_data();
    std::cout << std::endl;

    std::cout << "--------------y on cpu: --------------" << std::endl;
    y.print_data();
    std::cout << "--------------mask on cpu: --------------" << std::endl;
    mask.print_data();
    Tensor grad_x;
    maxpool_backward(&grad_y, &mask, &grad_x);
    std::cout << "--------------grad x on cpu: --------------" << std::endl;
    grad_x.print_data();
    std::cout << std::endl;

    x._gpu();
    grad_y._gpu();
    Tensor y_gpu = maxpool_forward(&x, &mask, K);
    std::cout << "--------------y on gpu: --------------" << std::endl;
    y_gpu.print_data();
    // y_gpu.print_information();
    std::cout << "--------------mask on gpu: --------------" << std::endl;
    mask.print_data();
    // mask.print_information();
    maxpool_backward(&grad_y, &mask, &grad_x);
    std::cout << "--------------grad x on gpu: --------------" << std::endl;
    grad_x.print_data();
    // grad_x.print_information();
}

void test_softmax_and_cel()
{
    int batch_size = 2;
    int class_num = 4;
    std::vector<int> in_shape = {batch_size, class_num};
    Tensor x(in_shape);
    for(int i = 0 ; i < x.get_element_num() ; i ++)
        x.get_data()[i] = i;
    Tensor y = softmax_forward(&x);
    std::cout << "--------------x on cpu: --------------" << std::endl;
    x.print_data();
    std::cout << "--------------y=softmax(x) on cpu: --------------" << std::endl;
    y.print_data();

    Tensor gt_prob(in_shape);
    fill_cpu(gt_prob.get_data(), 0.0, gt_prob.get_element_num());
    gt_prob.get_data()[0] = 0.4;
    gt_prob.get_data()[1] = 0.3;    
    gt_prob.get_data()[2] = 0.3;
    gt_prob.get_data()[5] = 0.5;
    gt_prob.get_data()[6] = 0.2;
    gt_prob.get_data()[7] = 0.3;
    std::cout << "--------------gt_prob on cpu: --------------" << std::endl;
    gt_prob.print_data();
    
    Tensor loss = cross_entropy_forward(&y, &gt_prob);
    std::cout << "--------------loss on cpu: --------------" << std::endl;
    loss.print_data();
    
    Tensor grad_x;
    cross_entropy_with_softmax_backward(&y, &gt_prob, &grad_x);
    std::cout << "--------------grad_x on cpu: --------------" << std::endl;
    grad_x.print_data();
    std::cout << std::endl;

    x._gpu();
    Tensor y_gpu = softmax_forward(&x);
    std::cout << "--------------y on gpu: --------------" << std::endl;
    y_gpu.print_data();
    Tensor gt_prob_gpu = gt_prob.gpu();
    std::cout << "--------------gt_prob on gpu: --------------" << std::endl;
    gt_prob_gpu.print_data();
    Tensor loss_gpu = cross_entropy_forward(&y_gpu, &gt_prob_gpu);
    std::cout << "--------------loss on gpu: --------------" << std::endl;
    loss_gpu.print_data();
    Tensor grad_x_gpu;
    cross_entropy_with_softmax_backward(&y_gpu, &gt_prob_gpu, &grad_x_gpu);
    std::cout << "--------------grad_x on gpu: --------------" << std::endl;
    grad_x_gpu.print_data();
    std::cout << std::endl;

}

int main()
{
    // test_initialize_tensor();
    // test_activate_func();
    // test_data_manager();
    // test_gemm();
    // test_fc();
    test_conv();
    // test_pooling();
    // test_softmax_and_cel(); 
    return 0;
}