#include "Tensor.h"

Tensor relu_forward(const Tensor* input);
Tensor relu_backward(const Tensor* forward_input, const Tensor* last_grad = NULL);

Tensor sigmoid_forward(const Tensor* input);
Tensor sigmoid_backward(const Tensor* forward_output, const Tensor* last_grad = NULL);

Tensor fc_forward(const Tensor* input, const Tensor* weight, const Tensor* bias);
void fc_backward(const Tensor* grad_y, const Tensor* input_x, const Tensor* weight, const Tensor* bias, 
                Tensor* grad_x, Tensor* grad_weight, Tensor* grad_bias);

Tensor conv2d_forward(const Tensor* input, const Tensor* weight, const Tensor* bias, int stride, int padding = 0);
void conv2d_backward(const Tensor* grad_y, const Tensor* input_x, const Tensor* weight, int stride, int padding,
                    Tensor* grad_x, Tensor* grad_weight, Tensor* grad_bias);

Tensor maxpool_forward(const Tensor* input, Tensor* mask, int K);
void maxpool_backward(const Tensor* grad_y, const Tensor* mask, Tensor* grad_x);

Tensor softmax_forward(const Tensor* input);

Tensor cross_entropy_forward(const Tensor* input, const Tensor* gt_prob);

void cross_entropy_with_softmax_backward(const Tensor* prob, const Tensor* gt_prob, Tensor* grad);