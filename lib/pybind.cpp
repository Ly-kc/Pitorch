#include "Tensor.h"
#include "tensor_func.h"

#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include <pybind11/numpy.h>

#include <boost/optional.hpp>

namespace py = pybind11;


//warp the functions above into python format
Tensor relu_forward_wrapper(const Tensor& input)
{
    return relu_forward(&input);
}

// Tensor relu_backward_wrapper(const Tensor& forward_input, boost::optional<Tensor&> last_grad)
// {
//     if(!last_grad.has_value())
//         return relu_backward(&forward_input);
//     else
//         return relu_backward(&forward_input, &(last_grad.value()));
// }

Tensor relu_backward_wrapper(const Tensor& forward_input, const Tensor& last_grad)
{
    return relu_backward(&forward_input, &last_grad);
}


Tensor sigmoid_forward_wrapper(const Tensor& input)
{
    return sigmoid_forward(&input);
}

Tensor sigmoid_backward_wrapper(const Tensor& forward_output, const Tensor& last_grad)
{
    return sigmoid_backward(&forward_output, &last_grad);
}

Tensor fc_forward_wrapper(const Tensor& input, const Tensor& weight, const Tensor& bias)
{
    return fc_forward(&input, &weight, &bias);
}

py::tuple fc_backward_wrapper(const Tensor& grad_y, const Tensor& input_x, const Tensor& weight, const Tensor& bias)
{
    Tensor grad_x, grad_weight, grad_bias;
    fc_backward(&grad_y, &input_x, &weight, &bias, &grad_x, &grad_weight, &grad_bias);
    return py::make_tuple(grad_x, grad_weight, grad_bias);
}

Tensor conv2d_forward_wrapper(const Tensor& input, const Tensor& weight, const Tensor& bias, int stride, int padding)
{
    return conv2d_forward(&input, &weight, &bias, stride, padding);
}

py::tuple conv2d_backward_wrapper(const Tensor& grad_y, const Tensor& input_x, const Tensor& weight, int stride, int padding)
{
    Tensor grad_x, grad_weight, grad_bias;
    conv2d_backward(&grad_y, &input_x, &weight, stride, padding, &grad_x, &grad_weight, &grad_bias);
    return py::make_tuple(grad_x, grad_weight, grad_bias);
}

py::tuple maxpool_forward_wrapper(const Tensor& input, int K)
{
    Tensor mask;
    Tensor output = maxpool_forward(&input, &mask, K);
    return py::make_tuple(output, mask);
}

Tensor maxpool_backward_wrapper(const Tensor& grad_y, const Tensor& mask)
{
    Tensor grad_x;
    maxpool_backward(&grad_y, &mask, &grad_x);
    return grad_x;
}

Tensor softmax_forward_wrapper(const Tensor& input)
{
    return softmax_forward(&input);
}

Tensor cross_entropy_forward_wrapper(const Tensor& input, const Tensor& gt_prob)
{
    return cross_entropy_forward(&input, &gt_prob);
}

Tensor cross_entropy_with_softmax_backward_wrapper(const Tensor& prob, const Tensor& gt_prob)
{
    Tensor grad;
    cross_entropy_with_softmax_backward(&prob, &gt_prob, &grad);
    return grad;
}


PYBIND11_MODULE(raw_tensor, m) {
    py::class_<Tensor>(m,"Tensor")
    // .def(py::init<std::vector<int>&, std::string>(), py::arg("shape"), py::arg("device")="cpu")  //confused with the one using numpy array. may be replaced by zeros()
    .def(py::init<std::vector<int>&, float, std::string>(), py::arg("shape"), py::arg("scalar"), py::arg("device")="cpu")
    .def(py::init<const py::array_t<float>& , std::string>(), py::arg("arr"), py::arg("device")="cpu")
    .def(py::init<const Tensor&>())
    .def(py::init<>())
    .def("numpy", &Tensor::to_numpy)
    .def("cpu", &Tensor::cpu)
    .def("gpu", &Tensor::gpu)
    .def("_cpu", &Tensor::_cpu)
    .def("_gpu", &Tensor::_gpu)
    .def("reshape", &Tensor::reshape)
    .def("shape", &Tensor::get_shape)
    .def("print_data", &Tensor::print_data)
    .def("print_information", &Tensor::print_information);
    // .def("__getitem__", &Tensor::at);

    m.def("ReLU_forward", &relu_forward_wrapper, "ReLU Module forward propagation function");
    // m.def("ReLU_backward", &relu_backward_wrapper, "ReLU Module backward propagation function", py::arg("forward_input"), py::arg("last_grad")=py::none());
    m.def("ReLU_backward", &relu_backward_wrapper, "ReLU Module backward propagation function", py::arg("forward_input"), py::arg("last_grad"));
    m.def("Sigmoid_forward", &sigmoid_forward_wrapper, "Sigmoid Module forward propagation function");
    m.def("Sigmoid_backward", &sigmoid_backward_wrapper, "Sigmoid Module backward propagation function", py::arg("forward_input"), py::arg("last_grad"));
    m.def("fc_forward", &fc_forward_wrapper, "FC Module forward propagation function");
    m.def("fc_backward", &fc_backward_wrapper, "FC Module backward propagation function");
    m.def("Convolution_forward", &conv2d_forward_wrapper, "Convolution Module forward propagation function");
    m.def("Convolution_backward", &conv2d_backward_wrapper, "Convolution Module backward propagation function");
    m.def("Pooling_forward", &maxpool_forward_wrapper, "Pooling Module forward propagation function");
    m.def("Pooling_backward", &maxpool_backward_wrapper, "Pooling Module backward propagation function");
    m.def("Softmax_forward", &softmax_forward_wrapper, "Softmax Module forward propagation function");
    m.def("Crossentropy_forward", &cross_entropy_forward_wrapper, "Crossentropy Module forward propagation function");
    m.def("Softmax_Crossentropy_backward", &cross_entropy_with_softmax_backward_wrapper, "Softmax & Crossentropy Module backward propagation function");

    }


