#pragma once

#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <iostream>
#include "gpu_func.h"
#include "cpu_func.h"
#include <assert.h>

class Tensor;

class Memory_Manager
{
    friend class Tensor;
    float* data;
    int* refcount;
    std::string device;
public:
    Memory_Manager();
    Memory_Manager(float* _data, std::string _device = "cpu");
    Memory_Manager(const Memory_Manager& other);
    ~Memory_Manager();
    void add_refcount();
    void sub_refcount();    //其实可以起个别名叫detach from data
    void reset(float* _data, std::string _device = "cpu");  //can only be used when the data is not used by anyone
    void operator=(const Memory_Manager& other); //can be used when the data is used by someone
};


class Tensor 
{
    //all the parameters are stored in the cpu, excpet for the data
    std::vector<int> shape;
    std::vector<int> strides;
    int ndim;
    int element_num;
    float* data; //need to be totally replaced by the Memory_Manager!!!!!!!!!!!!!!!!!!!!!!!
    std::string device;
    Memory_Manager manager;

public:
    Tensor();
    Tensor(std::vector<int>& _shape, std::string _device="cpu");
    Tensor(std::vector<int>& _shape, float* _data, std::string _device="cpu");
    Tensor(std::vector<int>& _shape, float scalar, std::string _device="cpu");
    Tensor(const pybind11::array_t<float>& arr, std::string _device="cpu");
    Tensor(const Tensor& other);   //shallow copy
    ~Tensor();

    pybind11::array_t<float> to_numpy() const;

    void _cpu();
    void _gpu();
    Tensor cpu() const;   //deep copy the tensor and move it to cpu               
    Tensor gpu() const;    //deep copy the tensor and move it to gpu

    Tensor reshape(std::vector<int> _shape);  //normally, the _shape is not so large, so just pass the copy as parameter
    void _reshape(std::vector<int> _shape);
    
    void fill(float scalar);

    Tensor squeeze(int axis);   //the tensor returned share the same data with the original one, but with different shape
    Tensor unsqueeze(int axis);     //the tensor returned share the same data with the original one, but with different shape
    Tensor& broadcast(const std::vector<int>& _shape);  //unfinished!!!
    void _squeeze(int axis);
    void _unsqueeze(int axis);

    Tensor& slice(std::vector<int> start, std::vector<int> end);
    Tensor& slice(std::vector<int> start, std::vector<int> end, std::vector<int> step);
    float& at(const std::vector<int>& index);
    Tensor copy() const;   //return deep copy of the tensor

    
    Tensor operator+(const Tensor& other);   //the result has its own data location
    Tensor operator-(const Tensor& other);
    Tensor operator*(const Tensor& other);
    Tensor operator/(const Tensor& other);
    Tensor operator+(float other);
    Tensor operator-(float other);
    Tensor operator*(float other);
    Tensor operator/(float other);
    
    void operator+=(const Tensor& other);
    void operator-=(const Tensor& other);
    void operator*=(const Tensor& other);
    void operator/=(const Tensor& other);
    void operator+=(float other);
    void operator-=(float other);
    void operator*=(float other);
    void operator/=(float other);

    //only for setting data value, not changing shape etc.
    void operator=(const Tensor& other);
    void operator=(float other);

    //become the shallow copy of another tensor
    void copy_from(const Tensor& other);

    void print_data() const;
    std::vector<int> get_shape() const;
    const std::vector<int>& get_strides() const;
    int get_ndim() const;
    int get_element_num() const;
    std::string get_device() const;
    void print_information() const;
    const float* read_data() const;
    float* get_data();
    int get_refcount() const;
};

