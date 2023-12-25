#pragma once
#include<iostream>
#include<cmath>

void add_cpu(float* dest, const float* src, int n);
void dec_cpu(float* dest, const float* src, int n);
void dot_cpu(float* dest, const float* src, int n);
void div_cpu(float* dest, const float* src, int n);

void add_cpu(float* dest, float scalar, int n);
void dec_cpu(float* dest, float scalar, int n);
void dot_cpu(float* dest, float scakar, int n);
void div_cpu(float* dest, float scalar, int n);

void fill_cpu(float* dest, float val, int n);
void copy_cpu(float* dest, float* src, int n);

//memory?
// void data_broadcast_cpu(float* dest, const float* src, std::vector<int> dest_shape, std::vector<int> src_shape);

void relu_cpu(float* dest, const float* src, int n);
void backward_relu_cpu(float* dest, const float* forward_input, int n);

void sigmoid_cpu(float* dest, const float* src, int n);
void backward_sigmoid_cpu(float* dest, const float* src, int n);

void gemm_cpu(bool transa, bool transb,const float* A,const float* B, float* C, int M, int N, int K, float alpha, float beta);
void batch_gemm_cpu(bool transa, bool transb,const float* A,const float* B, float* C, int batch_size, int M, int N, int K, float alpha, float beta);

void fc_forward_cpu(const float* input, float* output, const float* weights, const float* bias, int in_dim, int out_dim, int batch_Size);
void fc_backward_cpu(const float* grad_y, const float* input_x, const float* weights, const float* bias, 
        float* grad_x, float* grad_weights, float* grad_bias, 
        int in_dim, int out_dim, int batch_Size
);

void conv2d_forward_cpu(const float* input, float* output, const float* weights, const float* bias, int B, int C_in, int C_out, int H, int W, int K, int stride, int padding = 0);
void conv2d_backward_cpu(const float* grad_y, const float* input_x, const float* weights,
        float* grad_x, float* grad_weights, float* grad_bias, 
        int B, int C_in, int C_out, int H, int W, int K, int stride, int padding = 0
);

void maxpool_forward_cpu(const float* input, float* output, float* mask, int B, int C, int H, int W, int K);
void maxpool_backward_cpu(const float* grad_y, const float* mask, float* grad_x, int B, int C, int H, int W, int K);

void softmax_forward_cpu(const float* input, float* output, int N, int C);

void cross_entropy_forward_cpu(const float* input, const float* gt_prob, float* output, int N, int C);
void cross_entropy_with_softmax_backward_cpu(const float* prob, const float* gt_prob, float* grad, int N, int C);