#include "cpu_func.h"

#include <cstring>

#include <omp.h>

void power_cpu(float* dest, float src, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for(int i=0;i<n;i++)
    {
        dest[i]=pow(dest[i],src);
    }
}

void add_cpu(float* dest, const float* src, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] += src[i];
    }
}
void dec_cpu(float* dest, const float* src, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] -= src[i];
    }
}
void dot_cpu(float* dest, const float* src, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] *= src[i];
    }
}
void div_cpu(float* dest, const float* src, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] /= src[i];
    }
}

void add_cpu(float* dest, float scalar, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] += scalar;
    }
}

void dec_cpu(float* dest, float scalar, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] -= scalar;
    }
}

void dot_cpu(float* dest, float scalar, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] *= scalar;
    }
}
void div_cpu(float* dest, float scalar, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] /= scalar;
    }
}

void fill_cpu(float* dest, float val, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] = val;
    }
}

void copy_cpu(float* dest, float* src, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] = src[i];
    }
}



void relu_cpu(float* dest, const float* src, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] = src[i] > 0 ? src[i] : 0;
    }
}


void backward_relu_cpu(float* dest, const float* forward_input, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] = forward_input[i] > 0 ? 1 : 0;
    }
}


void sigmoid_cpu(float* dest, const float* src, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] = 1.0 / (1.0 + exp(-src[i]));
    }
}

void backward_sigmoid_cpu(float* dest, const float* src, int n)
{
    #pragma omp parallel for schedule(static) collapse(1)
    for (int i = 0; i < n; i++)
    {
        dest[i] = src[i] * (1 - src[i]);
    }
}

void gemm_cpu(bool transa, bool transb, const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta)
{
    //C = alpha* op(A) @ op(B) + beta*C
    //op(A)=M*K, op(B)=K*N, C=M*N
    #pragma omp parallel for schedule(static) collapse(2)
    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {
            float sum=0;
            for(int k=0;k<K;k++)
            {
                float element_a = transa ? A[k*M+i] : A[i*K+k];
                float element_b = transb ? B[j*K+k] : B[k*N+j];
                sum+=element_a*element_b;
                // std::cout <<i<<' '<<j<<' '<< element_a << " " << element_b << std::endl;
            }
            C[i*N+j]=alpha*sum+beta*C[i*N+j];
            // std::cout << sum <<std::endl;
        }
    }

}

void batch_gemm_cpu(bool transa, bool transb,const float* A,const float* B, float* C, int batch_size, int M, int N, int K, float alpha, float beta)
{
    //temporary support the same batch size for A B and C    
    std::cout << "batch_gemm_cpu not supported yet" << std::endl;
}

void fc_forward_cpu(const float* input, float* output, const float* weights, const float* bias, 
                    int in_dim, int out_dim, int batch_Size){
    //input: batch_size * in_dim
    //weights: out_dim * in_dim
    //bias: out_dim
    //output: batch_size * out_dim

    // (batch_size,in_dim) @ (in_dim,out_dim) = (batch_size,out_dim)   
    gemm_cpu(
        false,true,
        input,weights,output,
        batch_Size,out_dim,in_dim,
        1.0,0.0
    );
    //add bias  (batch_size,1) @ (1,out_dim) = (batch_size,out_dim)
    float* temp = new float[batch_Size];
    fill_cpu(temp,1.0,batch_Size);
    gemm_cpu(
        false,false,
        temp,bias,output,
        batch_Size,out_dim,1,
        1.0,1.0
    );
}

void fc_backward_cpu(const float* grad_y, const float* input_x, const float* weights, const float* bias, 
        float* grad_x, float* grad_weights, float* grad_bias, 
        int in_dim, int out_dim, int batch_Size){
    //grad_y: batch_size * out_dim
    //input_x: batch_size * in_dim
    //weights: out_dim * in_dim
    //bias: out_dim

    //grad_x = grad_y @ weights
    //(batch_size, out_dim) @ (out_dim, in_dim) = (batch_size, in_dim)
    gemm_cpu(
        false,false,
        grad_y,weights,grad_x,
        batch_Size,in_dim,out_dim,
        1.0,0.0
    );
    //grad_weights = grad_y.T @ input_x
    //(out_dim, batch_size) @ (batch_size, in_dim) = (out_dim, in_dim)
    gemm_cpu(
        true,false,
        grad_y,input_x,grad_weights,
        out_dim,in_dim,batch_Size,
        1.0,0.0
    );
    //grad_bias = grad_y.T @ 1
    //(out_dim, batch_size) @ (batch_size, 1) = (out_dim, 1)
    float* temp = new float[batch_Size];
    fill_cpu(temp,1.0,batch_Size);
    gemm_cpu(
        true,false,
        grad_y,temp,grad_bias,
        out_dim,1,batch_Size,
        1.0,0.0
    );
}

void conv2d_forward_cpu(const float* input, float* output, const float* weights, const float* bias, 
                        int B, int C_in, int C_out, int H, int W, int K, int stride, int padding){
    //input: B * C_in * H * W
    //weights: C_out * C_in * K * K
    //bias: C_out
    //output: B * C_out * H_out * W_out
    
    //conv with multi level loop
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    int in_stride1 = C_in * H * W,
        in_stride2 = H * W,
        in_stride3 = W;
    int out_stride1 = C_out * H_out * W_out,
        out_stride2 = H_out * W_out,
        out_stride3 = W_out;

    #pragma omp parallel for schedule(static) collapse(4)
    for(int b=0;b<B;b++)
        for(int c_out=0;c_out<C_out;c_out++)
            for(int h_out=0;h_out<H_out;h_out++)
                for(int w_out=0;w_out<W_out;w_out++)
                {
                    float sum=0;
                    for(int c_in=0;c_in<C_in;c_in++)
                        for(int k_h=0;k_h<K;k_h++)
                            for(int k_w=0;k_w<K;k_w++)
                            {
                                int h_in = h_out * stride + k_h - padding;
                                int w_in = w_out * stride + k_w - padding;
                                if(h_in>=0 && h_in<H && w_in>=0 && w_in<W)
                                {
                                    sum+=input[b*in_stride1+c_in*in_stride2+h_in*in_stride3+w_in]*weights[c_out*C_in*K*K+c_in*K*K+k_h*K+k_w];
                                }
                            }
                    output[b*out_stride1+c_out*out_stride2+h_out*out_stride3+w_out]=sum+bias[c_out];
                }
}
void conv2d_backward_cpu(const float* grad_y, const float* input_x, const float* weights,
        float* grad_x, float* grad_weights, float* grad_bias, 
        int B, int C_in, int C_out, int H, int W, int K, int stride, int padding){
    //grad_y: B * C_out * H_out * W_out
    //input_x, grad_x: B * C_in * H * W
    //weights, grad_weights: C_out * C_in * K * K
    //grad_bias: C_out
    int H_out = (H + 2 * padding - K) / stride + 1;
    int W_out = (W + 2 * padding - K) / stride + 1;
    int in_stride1 = C_in * H * W,
        in_stride2 = H * W,
        in_stride3 = W;
    int out_stride1 = C_out * H_out * W_out,
        out_stride2 = H_out * W_out,
        out_stride3 = W_out;
    //aggregate gradient in every place of x with multilevel loop
    
    #pragma omp parallel for schedule(static) collapse(2)
    for(int b=0;b<B;b++)
        for(int c_in=0;c_in<C_in;c_in++)
            for(int c_out=0;c_out<C_out;c_out++)
                for(int h_out=0;h_out<H_out;h_out++)
                    for(int w_out=0;w_out<W_out;w_out++)
                    {
                        for(int k_h=0;k_h<K;k_h++)
                            for(int k_w=0;k_w<K;k_w++)
                            {
                                int h_in = h_out * stride + k_h - padding;
                                int w_in = w_out * stride + k_w - padding;
                                if(h_in>=0 && h_in<H && w_in>=0 && w_in<W) {
                                    grad_x[b*in_stride1+c_in*in_stride2+h_in*in_stride3+w_in]+=
                                        grad_y[b*out_stride1+c_out*out_stride2+h_out*out_stride3+w_out]*weights[c_out*C_in*K*K+c_in*K*K+k_h*K+k_w];
                                }
                            }
                    }
    #pragma omp parallel for schedule(static) collapse(2)
    for(int c_out=0;c_out<C_out;c_out++)
        for(int c_in=0;c_in<C_in;c_in++)
            for(int b=0;b<B;b++)
                for(int h_out=0;h_out<H_out;h_out++)
                    for(int w_out=0;w_out<W_out;w_out++)
                    {
                        for(int k_h=0;k_h<K;k_h++)
                            for(int k_w=0;k_w<K;k_w++)
                            {
                                int h_in = h_out * stride + k_h - padding;
                                int w_in = w_out * stride + k_w - padding;
                                if(h_in>=0 && h_in<H && w_in>=0 && w_in<W)
                                {
                                    grad_weights[c_out*C_in*K*K+c_in*K*K+k_h*K+k_w]+=
                                        grad_y[b*out_stride1+c_out*out_stride2+h_out*out_stride3+w_out]*input_x[b*in_stride1+c_in*in_stride2+h_in*in_stride3+w_in];
                                }
                            }
                    }
    #pragma omp parallel for schedule(static) collapse(1)
    for(int c_out=0;c_out<C_out;c_out++)
        for(int b=0;b<B;b++)
            for(int h_out=0;h_out<H_out;h_out++)
                for(int w_out=0;w_out<W_out;w_out++)
                {
                    grad_bias[c_out]+=grad_y[b*out_stride1+c_out*out_stride2+h_out*out_stride3+w_out];
                }
}

void maxpool_forward_cpu(const float* input, float* output, float* mask, int B, int C, int H, int W, int stride)
{
    //input: B * C * H * W
    //output: B * C * H_out * W_out
    //mask: B * C * H * W
    int H_out = (H-1) / stride + 1;
    int W_out = (W-1) / stride + 1;
    int in_stride1 = C * H * W,
        in_stride2 = H * W,
        in_stride3 = W;
    int out_stride1 = C * H_out * W_out,
        out_stride2 = H_out * W_out,
        out_stride3 = W_out;

    memset(mask,0,sizeof(float)*B*C*H*W);
    #pragma omp parallel for schedule(static) collapse(4)
    for(int b=0;b<B;b++)
        for(int c=0;c<C;c++)
            for(int h_out=0;h_out<H_out;h_out++)
                for(int w_out=0;w_out<W_out;w_out++)
                {
                    float max=-1e10;
                    int max_index=-1;
                    for(int k_h=0;k_h<stride;k_h++)
                        for(int k_w=0;k_w<stride;k_w++)
                        {
                            int h_in = h_out * stride + k_h;
                            int w_in = w_out * stride + k_w;
                            if(h_in<H && w_in<W)
                            {
                                if(input[b*in_stride1+c*in_stride2+h_in*in_stride3+w_in]>max)
                                {
                                    max=input[b*in_stride1+c*in_stride2+h_in*in_stride3+w_in];
                                    max_index=h_in*in_stride3+w_in;
                                }
                            }
                        }
                    output[b*out_stride1+c*out_stride2+h_out*out_stride3+w_out]=max;
                    mask[b*in_stride1+c*in_stride2+max_index]=1.;
                }
}

void maxpool_backward_cpu(const float* grad_y, const float* mask, float* grad_x, int B, int C, int H, int W, int stride)
{
    //grad_y: B * C * H_out * W_out
    //grad_x, mask: B * C * H * W
    int H_out = (H-1) / stride + 1;
    int W_out = (W-1) / stride + 1;
    int in_stride1 = C * H * W,
        in_stride2 = H * W,
        in_stride3 = W;
    int out_stride1 = C * H_out * W_out,
        out_stride2 = H_out * W_out,
        out_stride3 = W_out;
    #pragma omp parallel for schedule(static) collapse(6)
    for(int b=0;b<B;b++)
        for(int c=0;c<C;c++)
            for(int h_out=0;h_out<H_out;h_out++)
                for(int w_out=0;w_out<W_out;w_out++)
                {
                    for(int k_h=0;k_h<stride;k_h++)
                        for(int k_w=0;k_w<stride;k_w++)
                        {
                            int h_in = h_out * stride + k_h;
                            int w_in = w_out * stride + k_w;
                            if(h_in<H && w_in<W)
                            {
                                if(mask[b*in_stride1+c*in_stride2+h_in*in_stride3+w_in] > 0)
                                {
                                    grad_x[b*in_stride1+c*in_stride2+h_in*in_stride3+w_in]=grad_y[b*out_stride1+c*out_stride2+h_out*out_stride3+w_out];
                                }
                            }
                        }
                }
}

void softmax_forward_cpu(const float* input, float* output, int N, int C)
{
    //input, output: N*C
    float max = -1e10;
    int element_num = N*C;
    for(int i = 0 ; i < element_num ; i++)
        if(max < input[i]) max = input[i];
    float esum[element_num] = {0};
    for(int i = 0 ; i < element_num ; i++)
    {
        output[i] = exp(input[i]-max);
        esum[i/C] += output[i];
    }
    #pragma omp parallel for schedule(static) collapse(1)
    for(int i = 0 ; i < element_num ; i++)
    {
        output[i] /= esum[i/C];
    }
}

void cross_entropy_forward_cpu(const float* pred_prob, const float* gt_prob, float* output, int N, int C)
{
    //input: N*C
    //gt_prob: N*C
    //output: 1
    float loss = 0;
    #pragma omp parallel for schedule(static) collapse(2)
    for(int i = 0 ; i < N ; i ++)
        for(int j = 0 ; j < C ; j ++)
        {
            float log_pred = log(pred_prob[i*C + j]);
            #pragma omp critical
            loss -= log_pred * gt_prob[i*C + j];
        }
    *output = loss/N;
}

void cross_entropy_with_softmax_backward_cpu(const float* pred_prob, const float* gt_prob, float* grad, int N, int C)
{
    //pred_prob, gt_prob, grad: N*C
    #pragma omp parallel for schedule(static) collapse(2)
    for(int i = 0 ; i < N ; i ++)
        for(int j = 0 ; j < C ; j ++)
        {
            grad[i*C + j] = (pred_prob[i*C + j] - gt_prob[i*C + j])/N;   //div N because the gradient of parameters in network will aggregate N times. Div it in the source;
        }
}