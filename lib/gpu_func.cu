#include "gpu_func.h"
#include "utils.h"
#define CHECK_CUDA_ERROR

__global__ void add_kernel(float* dest, const float* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] += src[i];
        i += stride;
    }
}
__global__ void dec_kernel(float* dest, const float* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] -= src[i];
        i += stride;
    }
}
__global__ void dot_kernel(float* dest, const float* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] *= src[i];
        i += stride;
    }
}
__global__ void div_kernel(float* dest, const float* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] /= src[i];
        i += stride;
    }
}
__global__ void add_kernel(float* dest, float val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] += val;
        i += stride;
    }
}
__global__ void dec_kernel(float* dest, float val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] -= val;
        i += stride;
    }
}
__global__ void dot_kernel(float* dest, float val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] *= val;
        i += stride;
    }
}
__global__ void div_kernel(float* dest, float val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] /= val;
        i += stride;
    }
}

__global__ void fill_kernel(float* dest, float val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] = val;
        i += stride;
    }
}

__global__ void relu_kernel(float* dest, const float* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] = src[i] > 0 ? src[i] : 0;
        i += stride;
    }
}

__global__ void backward_relu_kernel(float* dest, const float* forward_input, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        dest[i] = forward_input[i] > 0 ? 1 : 0;
        i += stride;
    }
}

__global__ void sigmoid_kernel(float* dest, const float* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float tmp;
    while (i < n)
    {
        tmp = __expf(-src[i]);
        dest[i] = 1 / (1 + tmp);
        i += stride;
    }
}

__global__ void backward_sigmoid_kernel(float* dest, const float* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < n)
    {
        dest[i] = src[i] * (1 - src[i]);
        i += stride;
    }
}

__global__ void im2col_kernel(const float* input, float* output, int B, int C, int H, int W, int K, int stride, int padding)
{
    //one thread to get one (C*K*K)
    //whole output: B*(H'*W')*(C*K*K)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride1 = blockDim.x * gridDim.x;
    int new_H = (H+2*padding-K)/stride + 1;
    int new_W = (W+2*padding-K)/stride + 1;

    int in_stride1 = C*H*W, 
        in_stride2 = H*W, 
        in_stride3 = W;
    int out_stride1 = new_H*new_W*C, 
        out_stride2 = new_W*C,
        out_stride3 = C; 
    
    while(i < B*C*new_H*new_W)
    {
        //coordinate in output map: (b,(h',w'),c)
        int b = i / out_stride1;
        int out_h = i % out_stride1 / out_stride2;
        int out_w = i % out_stride1 % out_stride2 / out_stride3;
        int out_c = i % out_stride1 % out_stride2 % out_stride3;
        
        for(int kh=0;kh<K;kh++)
            for(int kw=0;kw<K;kw++)
            {
                //coordinate in original image
                int img_h = out_h*stride - padding + kh;
                int img_w = out_w*stride - padding + kw;
                int out_index = i*K*K + kh*K + kw;   //coordinate in output map: (b,(h,w),(c,kh,kw))
                if(img_h<0 || img_h>=H || img_w<0 || img_w>=W){
                    output[out_index] = 0;
                }
                else
                {
                    //coordinate in original img: (b,c,img_h,img_w)
                    int input_index = b*in_stride1 + out_c*in_stride2 + img_h*in_stride3 + img_w;                        
                    output[out_index] = input[input_index];
                }
            }
        i += stride1;
    }

}

__global__ void col2im_kernel(const float* input, float* output, int B, int C, int H, int W, int K, int stride, int padding)
{
    //one thread take responsibility for one element in input
    //whole input: B*(H'*W')*(C*K*K)   imgcol
    //output: B*C*H*W  img
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride1 = blockDim.x * gridDim.x;
    int new_H = (H+2*padding-K)/stride + 1;
    int new_W = (W+2*padding-K)/stride + 1;

    int in_stride1 = new_H*new_W*C*K*K, 
        in_stride2 = new_W*C*K*K, 
        in_stride3 = C*K*K,
        in_stride4 = K*K;
    int out_stride1 = C*H*W,
        out_stride2 = H*W,
        out_stride3 = W;

    //aggregate the value of input imcol to output img with automicAdd
    while(i < B*new_H*new_W*C*K*K)
    {
        //coordinate in input map: (b,(h',w'),(c,kh,kw))
        int b = i / in_stride1;
        int h = i % in_stride1 / in_stride2;
        int w = i % in_stride1 % in_stride2 / in_stride3;
        int c = i % in_stride1 % in_stride2 % in_stride3 / in_stride4;
        int kh = i % in_stride1 % in_stride2 % in_stride3 % in_stride4 / K;
        int kw = i % in_stride1 % in_stride2 % in_stride3 % in_stride4 % K;
        //coordinate in original image
        int img_h = h*stride - padding + kh;
        int img_w = w*stride - padding + kw;
        if(img_h>=0 && img_h<H && img_w>=0 && img_w<W){
            //coordinate in original img: (b,c,img_h,img_w)
            int output_index = b*out_stride1 + c*out_stride2 + img_h*out_stride3 + img_w;
            atomicAdd(output+output_index, input[i]);
        }
        i += stride1;
    }
}

__global__ void maxpool_forward_kernel(const float* input, float* output, float* mask, int B, int C, int H, int W, int K)
{
    //input,mask: B * Cin * H * W
    //output: B * Cin * H' * W'
    //Each thread is responsible for a local window
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride1 = blockDim.x * gridDim.x;
    int outH = (H - 1)/K + 1;
    int outW = (W - 1)/K + 1;

    int out_strides1 = C*outH*outW, 
        out_strides2 = outH*outW, 
        out_strides3 = outW;
    int in_strides1 = C*H*W, 
        in_strides2 = H*W, 
        in_strides3 = W;
    while(i < B*C*outH*outW)
    {
        //coordinate in output map: (b,c,(wh,ww))
        int b = i / out_strides1;
        int c = i % out_strides1 / out_strides2;
        int oh = i % out_strides1 % out_strides2 / out_strides3;
        int ow = i % out_strides1 % out_strides2 % out_strides3;
        //coordinate of the start point in input map: (b,c,(h,w))
        int ih =oh * K;
        int iw =ow * K;

        float max_val = -1e10;
        int max_index = -1;
        int start_index = b*in_strides1 + c*in_strides2 + ih*in_strides3 + iw;
        for(int kh=0;kh<K;kh++){
            for(int kw=0;kw<K;kw++){
                //coordinate in input map: (b,c,(ih+kh,iw+kw))
                if(ih + kh >= H || iw + kw >= W) continue;
                int input_index = start_index + kh*in_strides3 + kw;
                float value = input[input_index];
                if(value > max_val){
                    max_val = value;
                    max_index = input_index;
                }
            }
        }
        output[i] = max_val;
        mask[max_index] = 1.;

        i += stride1;   
    }
}

__global__ void maxpool_backward_kernel(const float* grad_y,const float* mask, float* grad_x, int B, int C, int H, int W, int K)
{
    //one thread is responsible for one element in grad_x
    //grad_y: B * Cin * outH * outW'
    //grad_x, mask: B * Cin * H * W
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride1 = blockDim.x * gridDim.x;
    int outH = (H - 1)/K + 1;
    int outW = (W - 1)/K + 1;

    int x_stride1 = C*H*W, 
        x_stride2 = H*W, 
        x_stride3 = W;
    int y_stride1 = C*outH*outW,
        y_stride2 = outH*outW,
        y_stride3 = outW;
    while(i < B*C*H*W)
    {
        //coordinate in grad_x: (b,c,(h,w))
        int b = i / x_stride1;
        int c = i % x_stride1 / x_stride2;
        int xh = i % x_stride1 % x_stride2 / x_stride3;
        int xw = i % x_stride1 % x_stride2 % x_stride3;
        //coordinate in grad_y: (b,c,(oh,ow))
        int yh = xh / K;
        int yw = xw / K;
        int grad_y_index = b*y_stride1 + c*y_stride2 + yh*y_stride3 + yw;
        if(mask[i] > 0){
            grad_x[i] = grad_y[grad_y_index];
        }
        else{
            grad_x[i] = 0;
        }
        i += stride1;
    }
}

__global__ void simple_max_kernel(const float* input, float* batch_max, int N ,int C)
{
    //input: N*C 
    //batch_max: N
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < N)
    {
        float max = -1e10;
        for(int j = 0 ; j < C; j ++)
        {
            float value = input[i*C + j];
            if(max < value) max = value;
        }
        batch_max[i]=  max;
        i += stride;
    }
}

__global__ void decmax_and_exp_kernel(const float* input,const float* batch_max, float* output, int N, int C)
{
    //input, output: N*C
    //batch_max: N
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < N*C)
    {
        int n = i/C;
        float res = exp(input[i] - batch_max[n]);
        output[i] = res;
        i += stride;
    }
}

__global__ void simple_sum_kernel(const float* input,float* batch_sum, int N, int C)
{
    //input: N*C 
    //batch_sum: N
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < N)
    {
        float sum = 0;
        for(int j = 0 ; j < C; j ++)
        {
            float value = input[i*C + j];
            sum += value;
        }
        batch_sum[i]=  sum;
        i += stride;
    }
}

__global__ void broadcast_div_kernel(float* output, const float* batch_sum, int N, int C)
{
    //input: N*C 
    //batch_sum: N
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < N*C)
    {
        output[i] /= batch_sum[i/C];
        i += stride;
    }
}

__global__ void cross_entropy_forward_kernel(const float* pred_prob, const float* gt_prob, float* temp, int N, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < N*C)
    {
        temp[i] = -gt_prob[i] * logf(pred_prob[i]); //自然对数
        // temp[i] = -gt_prob[i] * logf(pred_prob[i]) / logf(2);
        i += stride;
    }
}

__global__ void cross_entropy_with_softmax_backward_kernel(const float* pred_prob, const float* gt_prob, float* grad, int N, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(i < N*C)
    {
        grad[i] = (pred_prob[i] - gt_prob[i])/N;
        i += stride;
    }
}

void add_gpu(float* dest, const float* src, int n)
{
    int BlockNum = CudaGetBlocks(n);
    add_kernel<<<BlockNum, BlockSize>>>(dest, src, n);
    sync_and_check_cuda_error(); 
}
void dec_gpu(float* dest, const float* src, int n)
{
    int BlockNum = CudaGetBlocks(n);
    dec_kernel<<<BlockNum, BlockSize>>>(dest, src, n);
    sync_and_check_cuda_error(); 
}
void dot_gpu(float* dest, const float* src, int n)
{
    int BlockNum = CudaGetBlocks(n);
    dot_kernel<<<BlockNum, BlockSize>>>(dest, src, n);
    sync_and_check_cuda_error(); 
}
void div_gpu(float* dest, const float* src, int n)
{
    int BlockNum = CudaGetBlocks(n);
    div_kernel<<<BlockNum, BlockSize>>>(dest, src, n);
    sync_and_check_cuda_error(); 
}
void add_gpu(float* dest, float val, int n)
{
    int BlockNum = CudaGetBlocks(n);
    add_kernel<<<BlockNum, BlockSize>>>(dest, val, n);
    sync_and_check_cuda_error(); 
}
void dec_gpu(float* dest, float val, int n)
{
    int BlockNum = CudaGetBlocks(n);
    dec_kernel<<<BlockNum, BlockSize>>>(dest, val, n);
    sync_and_check_cuda_error(); 
}
void dot_gpu(float* dest, float val, int n)
{
    int BlockNum = CudaGetBlocks(n);
    dot_kernel<<<BlockNum, BlockSize>>>(dest, val, n);
    sync_and_check_cuda_error(); 
}
void div_gpu(float* dest, float val, int n)
{
    int BlockNum = CudaGetBlocks(n);
    div_kernel<<<BlockNum, BlockSize>>>(dest, val, n);
    sync_and_check_cuda_error(); 
}

void cout_gpu(const float* input, int n){
    float* temp = new float[n];
    cudaMemcpy(temp, input, n * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0;i<n;i++){
        std::cout<<temp[i]<<" ";
    }
    std::cout<<std::endl;
    sync_and_check_cuda_error(); 
}

void fill_gpu(float* dest, float val, int n)
{
    int BlockNum = CudaGetBlocks(n);
    fill_kernel<<<BlockNum, BlockSize>>>(dest, val, n);
    sync_and_check_cuda_error();
}

void copy_gpu(float* dest, const float* src, int n)
{
    cudaMemcpy(dest, src, n * sizeof(float), cudaMemcpyDeviceToDevice);
    sync_and_check_cuda_error();
}



void relu_gpu(float* dest, const float* src, int n)
{
    int BlockNum = CudaGetBlocks(n);
    relu_kernel<<<BlockNum, BlockSize>>>(dest, src, n);
    sync_and_check_cuda_error();
}

void backward_relu_gpu(float* dest,const float* forward_input, int n)
{
    int BlockNum = CudaGetBlocks(n);
    backward_relu_kernel<<<BlockNum, BlockSize>>>(dest, forward_input, n);
    sync_and_check_cuda_error();
}


void sigmoid_gpu(float* dest, const float* src, int n)
{
    int BlockNum = CudaGetBlocks(n);
    sigmoid_kernel<<<BlockNum, BlockSize>>>(dest, src, n);
    sync_and_check_cuda_error();
}

void backward_sigmoid_gpu(float* dest, const float* src, int n)
{
    int BlockNum = CudaGetBlocks(n);
    backward_sigmoid_kernel<<<BlockNum, BlockSize>>>(dest, src, n);
    sync_and_check_cuda_error(); 
}

void gemm_gpu(bool transa, bool transb, const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta)
{
    //we use mA to represent the matrix in math explained in row-major
    //our goal: mC = alpha* op(mA) @ op(mB) + beta*mC
    //dimension in math: op(mA)=M*K, op(mB)=K*N, mC=M*N
    //that means, when transa=true, mA is K*M, else mA is M*K

    //below, A,B,C are considered to be column-major
    //if we use cmA to represent the matrix in math explained in column-major, then cmA=mA.T 
    //The leading dimension of A is the number of rows of cmA, which is the number of columns of mA
    cublasStatus_t status;
    cublasSgemm(
        cublasHandle(),
        transb ? CUBLAS_OP_T : CUBLAS_OP_N,
        transa ? CUBLAS_OP_T : CUBLAS_OP_N,
        N, M, K,
        &alpha, 
        B, transb ? K : N,
        A, transa ? M : K,
        &beta,
        C, N
    );

    sync_and_check_cuda_error(); 
}

void batch_gemm_gpu(bool transa, bool transb, const float* A, const float* B, float* C, 
                    int M, int N, int K, float alpha, float beta, 
                    bool ba,bool bb, bool bc, int batch_size){
    cublasStatus_t status;
    // cublas initialize
    // std::cout << "batch gemm start" <<M<<' '<<N<<' '<<K<<' '<<transa<<' '<<transb << ba << bb << bc  << batch_size<< std::endl;
    // cout_gpu(A,M*K*100);
    // cout_gpu(B,N*K);
    // cout_gpu(C,M*N);
    status = cublasSgemmStridedBatched(
        cublasHandle(),
        transb ? CUBLAS_OP_T : CUBLAS_OP_N,
        transa ? CUBLAS_OP_T : CUBLAS_OP_N,
        N, M, K,
        &alpha, 
        B, transb ? K : N, bb ? K*N : 0,
        A, transa ? M : K, ba ? K*M : 0,
        &beta,
        C, N, M*N,
        batch_size
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "batch gemm error" << std::endl;
        std::cout << status << std::endl;
        exit(0);
    }
    // cout_gpu(C,M*N);
    sync_and_check_cuda_error(); 
}

void fc_forward_gpu(const float* input, float* output, const float* weights, const float* bias, 
                    int in_dim, int out_dim, int batch_Size){
    //input: batch_size * in_dim
    //weights: out_dim * in_dim
    //bias: out_dim
    //output: batch_size * out_dim

    // (batch_size,in_dim) @ (in_dim,out_dim) = (batch_size,out_dim)   
    gemm_gpu(
        false,true,
        input,weights,output,
        batch_Size,out_dim,in_dim,
        1.0,0.0
    );
    //cout input
    // cout_gpu(input, batch_Size*in_dim);
    //add bias  (batch_size,1) @ (1,out_dim) = (batch_size,out_dim)
    float* temp;
    cudaMalloc((void**)&temp, batch_Size*out_dim*sizeof(float));
    fill_gpu(temp,1.0,batch_Size);
    gemm_gpu(
        false,false,
        temp,bias,output,
        batch_Size,out_dim,1,
        1.0,1.0
    );
    //free temp
    cudaFree(temp);
    sync_and_check_cuda_error(); 
}

void fc_backward_gpu(const float* grad_y, const float* input_x, const float* weights, const float* bias, 
        float* grad_x, float* grad_weights, float* grad_bias, 
        int in_dim, int out_dim, int batch_Size){
    //grad_y: batch_size * out_dim
    //input_x: batch_size * in_dim
    //weights: out_dim * in_dim
    //bias: out_dim

    //grad_x = grad_y @ weights
    //(batch_size, out_dim) @ (out_dim, in_dim) = (batch_size, in_dim)
    gemm_gpu(
        false,false,
        grad_y,weights,grad_x,
        batch_Size,in_dim,out_dim,
        1.0,0.0
    );
    //grad_weights = grad_y.T @ input_x
    //(out_dim, batch_size) @ (batch_size, in_dim) = (out_dim, in_dim)
    gemm_gpu(
        true,false,
        grad_y,input_x,grad_weights,
        out_dim,in_dim,batch_Size,
        1.0,0.0
    );
    //grad_bias = grad_y.T @ 1
    //(out_dim, batch_size) @ (batch_size, 1) = (out_dim, 1)
    float* temp;
    cudaMalloc((void**)&temp, batch_Size*sizeof(float));

    fill_gpu(temp,1.0,batch_Size);
    gemm_gpu(
        true,false,
        grad_y,temp,grad_bias,
        out_dim,1,batch_Size,
        1.0,0.0
    );
    
    cudaFree(temp);
    sync_and_check_cuda_error(); 
}

void im2col_gpu(const float* input, float* output, int B, int C, int H, int W, int K, int stride, int padding)
{
    //B can equal to 1
    //default zero padding
    //input: B*C*H*W
    //H' = (H+2*padding-K)/stride + 1
    //output: B*(H'*W')*(C*K*K)
    int new_H = (H+2*padding-K)/stride + 1;
    int new_W = (W+2*padding-K)/stride + 1;
    int BlockNum = CudaGetBlocks(B*C*new_H*new_W);
    im2col_kernel<<<BlockNum, BlockSize>>>(input, output, B, C, H, W, K, stride, padding);
    // cout_gpu(output, B*C*new_H*new_W*K*K);
    sync_and_check_cuda_error();    
}

void col2im_gpu(const float* input, float* output, int B, int C, int H, int W, int K, int stride, int padding)
{
    //aggregate the value of input imcol to output img
    //B can equal to 1
    //default zero padding
    //input: B*(H'*W')*(C*K*K)
    //H' = (H+2*padding-K)/stride + 1
    //output: B*C*H*W
    int new_H = (H+2*padding-K)/stride + 1;
    int new_W = (W+2*padding-K)/stride + 1;
    int BlockNum = CudaGetBlocks(B*new_H*new_W*C*K*K);
    col2im_kernel<<<BlockNum, BlockSize>>>(input, output, B, C, H, W, K, stride, padding);
    sync_and_check_cuda_error();    

}


void conv2d_forward_gpu(const float* input, float* output, const float* weights, const float* bias, int B, int C_in, int C_out, int H, int W, int K, int stride, int padding)
{
    //input: B*Cin*H*W
    //weights: Cout*Cin*K*K
    //bias: Cout
    //H' = (H+2*padding-K)/stride + 1
    //W' = (W+2*padding-K)/stride + 1
    //imgcol: B*(H'*W')*(C*K*K)
    //output: B*Cout*H'*W'
    float* imgcol;
    int outH = (H+2*padding-K)/stride + 1;
    int outW = (W+2*padding-K)/stride + 1;
    cudaMalloc((void**)&imgcol, B*outH*outW*C_in*K*K*sizeof(float));
    im2col_gpu(input, imgcol, B, C_in, H, W, K, stride, padding);
    
    //output = weights @ imgcol.T + bias   Here the transepose exclude the 0th dim  (Cout,(Cin,K,K)) @ B((H',W'),(Cin,K,K)).T = B(Cout,H',W')
    batch_gemm_gpu(
        false,true,
        weights,imgcol,output,
        C_out,outH*outW,C_in*K*K,
        1.0,0.0,
        false,true,true,
        B
    );
    //add bias   (Cout,1) @ (1,H'*W') = (Cout,H'*W')     (Cout,H'*W') + B(Cout,H'*W') = B(Cout,H'*W')
    float* temp;
    cudaMalloc((void**)&temp, outH*outW*sizeof(float));
    fill_gpu(temp,1.0,outH*outW);
    batch_gemm_gpu(
        false,false,
        bias,temp,output,
        C_out,outH*outW,1,
        1.0,1.0,
        false,false,true,
        B
    );

    //free imgcol and temp
    cudaFree(imgcol);
    cudaFree(temp);
    sync_and_check_cuda_error();
}


void conv2d_backward_gpu(const float* grad_y, const float* input_x, const float* weights,
        float* grad_x, float* grad_weights, float* grad_bias, 
        int B, int C_in, int C_out, int H, int W, int K, int stride, int padding){
    //grad_y: B*Cout*H'*W'
    //input_x: B*Cin*H*W
    //weights: Cout*Cin*K*K
    //grad_x: B*Cin*H*W
    //grad_weights: Cout*Cin*K*K
    //grad_bias: Cout
    
    //imgcol: B*(H'*W')*(Cin*K*K)
    float* imgcol;
    int outH = (H+2*padding-K)/stride + 1;
    int outW = (W+2*padding-K)/stride + 1;
    cudaMalloc((void**)&imgcol, B*outH*outW*C_in*K*K*sizeof(float));
    im2col_gpu(input_x, imgcol, B, C_in, H, W, K, stride, padding);

    //y = weights @ imgcolx.T + bias   Here the transepose exclude the 0th dim  (Cout,(Cin,K,K)) @ B((H',W'),(Cin,K,K)).T = B(Cout,H',W')
    //grad_weights = grad_y @ imgcolx
    float* batch_grad_weights; //B*Cout*Cin*K*K
    cudaMalloc((void**)&batch_grad_weights, B*C_out*C_in*K*K*sizeof(float));
    batch_gemm_gpu(
        false,false,
        grad_y, imgcol, batch_grad_weights,
        C_out,C_in*K*K,outH*outW,
        1.0,0.0,
        true,true,true,
        B
    );
    //get grad_weights
    //(1,B) @ (B,Cout,Cin,K,K) = (1,Cout,Cin,K,K)
    float* tempB; 
    cudaMalloc((void**)&tempB, B*sizeof(float)); //B
    fill_gpu(tempB,1.0,B);
    gemm_gpu(
        false,false,
        tempB, batch_grad_weights, grad_weights,
        1,C_out*C_in*K*K,B,
        1.0,0.0
    );
    //to get bias of shape cout from grad_y of shape (B,Cout,H',W'), we need to sum over B,H',W':
    //B*Cout*H'*W' @ (H'*W', 1) = B*Cout   (1,B)@(B,Cout) = (1,Cout)   
    float* batch_grad_bias; //B*Cout
    cudaMalloc((void**)&batch_grad_bias, B*C_out*sizeof(float));
    float* temp_outHW;
    cudaMalloc((void**)&temp_outHW, outH*outW*sizeof(float));
    fill_gpu(temp_outHW,1.0,outH*outW);
    batch_gemm_gpu(
        false,false,
        grad_y, temp_outHW, batch_grad_bias,
        C_out,1,outH*outW,
        1.0,0.0,
        true,false,true,
        B
    );
    gemm_gpu(
        false,false,
        tempB, batch_grad_bias, grad_bias,
        1,C_out,B,
        1.0,0.0
    );
    
    //get grad_imgcolx
    //grad_imgcolx = (weights.T @ grad_y).T = grad_y.T @ weights  (where the transpose exclude the 0th(batch) dim)
    //(H'*W',Cin*K*K) = (H'*W', cout) @ (cout, Cin*K*K)
    float* grad_imgcolx;
    cudaMalloc((void**)&grad_imgcolx, B*outH*outW*C_in*K*K*sizeof(float));
    batch_gemm_gpu(
        true,false,
        grad_y, weights, grad_imgcolx,
        outH*outW,C_in*K*K,C_out,
        1.0,0.0,
        true,false,true,
        B
    );
    // cout_gpu(grad_imgcolx, B*outH*outW*C_in*K*K);
    //get grad_x by col2img
    col2im_gpu(grad_imgcolx, grad_x, B, C_in, H, W, K, stride, padding);
    
    //free any malloced
    cudaFree(imgcol);
    cudaFree(batch_grad_weights);
    cudaFree(tempB);
    cudaFree(batch_grad_bias);
    cudaFree(temp_outHW);
    cudaFree(grad_imgcolx);
    sync_and_check_cuda_error();
}

void maxpool_forward_gpu(const float* input, float* output, float* mask, int B, int C, int H, int W, int K)
{
    //input: B * Cin * H * W
    //output: B * Cin * H' * W'
    //mask : B * Cin * H * W
    int outH = (H - 1)/K + 1;
    int outW = (W - 1)/K + 1;
    cudaMemset(mask, 0, B*C*H*W*sizeof(float));
    int BlockNum = CudaGetBlocks(B*C*outH*outW);
    maxpool_forward_kernel<<<BlockNum, BlockSize>>>(input, output, mask, B, C, H, W, K);

    sync_and_check_cuda_error();
}

void maxpool_backward_gpu(const float* grad_y, const float* mask, float* grad_x, int B, int C, int H, int W, int K)
{
    //grad_y: B * Cin * H' * W'
    //mask : B * Cin * H * W
    //grad_x: B * Cin * H * W
    fill_gpu(grad_x, 0.0, B*C*H*W);
    
    int BlockNum = CudaGetBlocks(B*C*H*W);
    maxpool_backward_kernel<<<BlockNum, BlockSize>>>(grad_y, mask, grad_x, B, C, H, W, K);

    sync_and_check_cuda_error();
}

void softmax_forward_gpu(const float* input, float* output, int N, int C)
{
    //input, output: N*C
    //get max among C
    float* batch_max;
    cudaMalloc((void**)&batch_max, N*sizeof(float));
    int BlockNum = CudaGetBlocks(N);
    simple_max_kernel<<<BlockNum, BlockSize>>>(input, batch_max, N ,C);
    BlockNum = CudaGetBlocks(N*C);
    decmax_and_exp_kernel<<<BlockNum, BlockSize>>>(input, batch_max, output, N, C);
    float* batch_expsum;
    cudaMalloc((void**)&batch_expsum, N*sizeof(float));
    BlockNum = CudaGetBlocks(N);
    simple_sum_kernel<<<BlockNum, BlockSize>>>(output, batch_expsum, N, C);
    BlockNum = CudaGetBlocks(N*C);
    broadcast_div_kernel<<<BlockNum, BlockSize>>>(output, batch_expsum, N, C);


    cudaFree(batch_max);
    cudaFree(batch_expsum);
    sync_and_check_cuda_error();
}

void cross_entropy_forward_gpu(const float* pred_prob, const float* gt_prob, float* loss, int N, int C)
{
    //input: N*C
    //gt_prob: N*C
    //output: 1
    int BlockNum = CudaGetBlocks(N*C);
    float* temp;
    cudaMalloc(&temp, N*C*sizeof(float));
    cross_entropy_forward_kernel<<<BlockNum, BlockSize>>>(pred_prob, gt_prob, temp, N, C);
    //sum temp with cublas
    cublasStatus_t status;

    sync_and_check_cuda_error_force();
    cublasSetPointerMode(cublasHandle(),CUBLAS_POINTER_MODE_DEVICE); 
    status = cublasSasum(cublasHandle(), N*C, temp, 1, loss);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasSasum failed" << int(status) << std::endl;
    }

    div_gpu(loss, N, 1);
    cudaFree(temp);
    sync_and_check_cuda_error();

}

void cross_entropy_with_softmax_backward_gpu(const float* pred_prob, const float* gt_prob, float* grad, int N, int C)
{
    //pred_prob, gt_prob, grad: N*C
    int BlockNum = CudaGetBlocks(N*C);
    cross_entropy_with_softmax_backward_kernel<<<BlockNum, BlockSize>>>(pred_prob, gt_prob, grad, N, C);

    sync_and_check_cuda_error();
}