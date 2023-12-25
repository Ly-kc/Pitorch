#ifndef CUDA_H
#define CUDA_H

#include <cublas_v2.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cmath>
#include <iostream>


const int BlockSize = 256;

inline int CudaGetBlocks(const int N)
{
    return (N + BlockSize - 1) / BlockSize;
} 


class CublasHandleCreator
{
    cublasHandle_t cublas_handle;
public:
    CublasHandleCreator()
    {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if(status != CUBLAS_STATUS_SUCCESS){
                std::cout << "cublas creation error1" << std::endl;
                std::cout << status << std::endl;
                exit(0);
        }
    }
    ~CublasHandleCreator()
    {
        cublasStatus_t status = cublasDestroy(cublas_handle);
        if(status != CUBLAS_STATUS_SUCCESS){
                std::cout << "cublas destroy error1" << std::endl;
                std::cout << status << std::endl;
                exit(0);
        }
    }
    cublasHandle_t& operator()()
    {
        return cublas_handle;
    }
};

static CublasHandleCreator cublasHandle;


void cout_gpu(const float* src, int n);

void add_gpu(float* dest, const float* src, int n);
void dec_gpu(float* dest, const float* src, int n);
void dot_gpu(float* dest, const float* src, int n);
void div_gpu(float* dest, const float* src, int n);

void add_gpu(float* dest, float scalar, int n);
void dec_gpu(float* dest, float scalar, int n);
void dot_gpu(float* dest, float scakar, int n);
void div_gpu(float* dest, float scalar, int n);

void fill_gpu(float* dest, float val, int n);
void copy_gpu(float* dest, const float* src, int n);

// void data_broadcast_gpu(float* dest, const float* src, std::vector<int> dest_shape, std::vector<int> src_shape);

void relu_gpu(float* dest, const float* src, int n);
void backward_relu_gpu(float* dest,const float* forward_input, int n);

void sigmoid_gpu(float* dest, const float* src, int n);
void backward_sigmoid_gpu(float* dest, const float* src, int n);

void gemm_gpu(bool transa, bool transb, const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta);
void batch_gemm_gpu(bool transa, bool transb, const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta, bool ba,bool bb, bool bc, int batch_size);

void fc_forward_gpu(const float* input, float* output, const float* weights, const float* bias, int in_dim, int out_dim, int batch_Size);
void fc_backward_gpu(const float* grad_y, const float* input_x, const float* weights, const float* bias, 
        float* grad_x, float* grad_weights, float* grad_bias, 
        int in_dim, int out_dim, int batch_Size
);

void im2col_gpu(const float* input, float* output, int B, int C, int H, int W, int K, int stride, int padding);
void col2im_gpu(const float* input, float* output, int B, int C, int H, int W, int K, int stride, int padding);

void conv2d_forward_gpu(const float* input, float* output, const float* weights, const float* bias, int B, int C_in, int C_out, int H, int W, int K, int stride, int padding = 0);
void conv2d_backward_gpu(const float* grad_y, const float* input_x, const float* weights,
        float* grad_x, float* grad_weights, float* grad_bias, 
        int B, int C_in, int C_out, int H, int W, int K, int stride, int padding = 0
);

void maxpool_forward_gpu(const float* input, float* output, float* mask, int B, int C, int H, int W, int K);
void maxpool_backward_gpu(const float* grad_y, const float* mask, float* grad_x, int B, int C, int H, int W, int K);

void softmax_forward_gpu(const float* input, float* output, int N, int C);

void cross_entropy_forward_gpu(const float* input, const float* gt_prob, float* output, int N, int C);
void cross_entropy_with_softmax_backward_gpu(const float* prob, const float* gt_prob, float* grad, int N, int C);


#endif