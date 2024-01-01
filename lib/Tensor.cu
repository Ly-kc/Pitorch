#include "Tensor.h"

Memory_Manager::Memory_Manager()
{
    data = NULL;
    refcount = NULL;
    device = "None";
}
Memory_Manager::Memory_Manager(float* _data, std::string _device)
{
    data = _data;
    refcount = new int(1);
    *refcount = 1;
    device = _device;
    if(device != "cpu" && device != "gpu")
    {
        std::cout << "device error" << std::endl;
        exit(0);
    }
}
Memory_Manager::Memory_Manager(const Memory_Manager& other)
{
    data = other.data;
    refcount = other.refcount;
    *refcount += 1;
    device = other.device;
}
Memory_Manager::~Memory_Manager()
{
    sub_refcount();
}
void Memory_Manager::add_refcount()
{
    *refcount += 1;
}
void Memory_Manager::sub_refcount()
{
    if(device == "None")
        return;
    *refcount -= 1;
    if(*refcount == 0)
    {
        if(device == "cpu")
        {
            delete[] data;
            delete refcount;
        }
        else if(device == "gpu")
        {
            CHECK(cudaFreeAsync(data, cudaStreamDefault));
            delete refcount;
        }
        // std::cout << "delete data at "<< data << std::endl;
    }
}    
//can only be used when the data is not used by anyone
void Memory_Manager::reset(float* _data, std::string _device)
{
    sub_refcount();
    data = _data;
    refcount = new int(1);
    *refcount = 1;
    device = _device;
}
//shallow copy
//can be used when the data is used by someone
void Memory_Manager::operator=(const Memory_Manager& other)
{
    sub_refcount();
    data = other.data;
    refcount = other.refcount;
    *refcount += 1;
    device = other.device;
}



Tensor::Tensor()
{
    device = "None";
    data = NULL;
}

Tensor::Tensor(std::vector<int>& _shape, std::string _device):shape(_shape), device(_device)
{
    // std::cout << "construct 1" << std::endl;
    ndim = shape.size();
    element_num = 1;
    for(int i = 0 ; i < ndim ; i ++)
    {
        element_num *= shape[i];
    }
    
    strides = std::vector<int>(ndim, 0);
    int stride = 1;
    for(int i = 0 ; i < ndim ; i ++)
    {
        strides[ndim-i-1] = stride;
        stride *= shape[ndim-i-1];
    }
    
    device = _device;
    if(device == "cpu")
    {
        data = new float[element_num];
        fill_cpu(data, 0, element_num);
    }
    else if(device == "gpu")
    {
        CHECK(cudaMallocAsync((void**)&data, element_num*sizeof(float), cudaStreamDefault));
        fill_gpu(data, 0, element_num);
        sync_and_check_cuda_error();
    }
    else
    {
        std::cout << "device error 1 when initialize tensor" << std::endl;
    }

    manager.reset(data, device);
}

Tensor::Tensor(std::vector<int>& _shape, float* _data, std::string _device):shape(_shape), device(_device)
{
    ndim = shape.size();
    element_num = 1;
    for(int i = 0 ; i < ndim ; i ++)
    {
        element_num *= shape[i];
    }
    
    strides = std::vector<int>(ndim, 0);
    int stride = 1;
    for(int i = 0 ; i < ndim ; i ++)
    {
        strides[ndim-i-1] = stride;
        stride *= shape[ndim-i-1];
    }
    
    if(device == "cpu")
    {
        data = new float[element_num];
        memcpy(data, _data, element_num*sizeof(float));

    }
    else if(device == "gpu")
    {
        CHECK(cudaMallocAsync((void**)&data, element_num*sizeof(float), cudaStreamDefault, cudaStreamDefault));
        CHECK(cudaMemcpy(data, _data, element_num*sizeof(float), cudaMemcpyHostToDevice));
    }
    else
    {
        std::cout << "device error 2 when initialize tensor: " << device << std::endl;
    }

    manager.reset(data, device);
}

Tensor::Tensor(std::vector<int>& _shape, float scalar, std::string _device):shape(_shape), device(_device)
{
    ndim = shape.size();
    element_num = 1;
    for(int i = 0 ; i < ndim ; i ++)
    {
        element_num *= shape[i];
    }
    
    strides = std::vector<int>(ndim, 0);
    int stride = 1;
    for(int i = 0 ; i < ndim ; i ++)
    {
        strides[ndim-i-1] = stride;
        stride *= shape[ndim-i-1];
    }
    
    if(device == "cpu")
    {
        data = new float[element_num];
        for(int i = 0 ; i < element_num ; i ++)
        {
            data[i] = scalar;
        }
    }
    else if(device == "gpu")
    {
        CHECK(cudaMallocAsync((void**)&data, element_num*sizeof(float), cudaStreamDefault));
        fill_gpu(data, scalar, element_num);
        sync_and_check_cuda_error();
    }
    else
    {
        std::cout << "device error 3 when initialize tensor" << std::endl;
    }
 
    manager.reset(data, device);
}

Tensor::Tensor(const pybind11::array_t<float>& arr, std::string _device)
{
    device = _device;
    shape = std::vector<int>(arr.ndim());
    for(int i = 0 ; i < arr.ndim() ; i ++)
    {
        shape[i] = arr.shape(i);
    }
    ndim = arr.ndim();
    element_num = arr.size();
    strides = std::vector<int>(ndim, 0);
    int stride = 1;

    for(int i = 0 ; i < ndim ; i ++)
    {
        strides[ndim-i-1] = stride;
        stride *= shape[ndim-i-1];
    }
    
    if(device == "cpu")
    {
        data = new float[arr.size()];
        memcpy(data, arr.data(), arr.size()*sizeof(float));
    }
    else if(device == "gpu")
    {
        CHECK(cudaMallocAsync((void**)&data, arr.size()*sizeof(float), cudaStreamDefault));
        CHECK(cudaMemcpy(data, arr.data(), arr.size()*sizeof(float), cudaMemcpyHostToDevice));
    }
    else
    {
        std::cout << "device error x when initialize tensor" << std::endl;
    }

    manager.reset(data, device);
}

//copy constructor(shallow copy)
Tensor::Tensor(const Tensor& other) 
{
    shape = other.shape;
    ndim = other.ndim;
    strides = other.strides;
    element_num = other.element_num;
    device = other.device;
    data = other.data;
    manager = other.manager;
}

Tensor::~Tensor()
{
    //we hope the manager can call its destructor automatically
}

pybind11::array_t<float> Tensor::to_numpy() const
{
    assert(device == "cpu" || device == "gpu");
    pybind11::array_t<float> arr(shape);
    float* ptr = arr.mutable_data();
    if(device == "cpu")
    {
        memcpy(ptr, data, element_num*sizeof(float));
    }
    else if(device == "gpu")
    {
        CHECK(cudaMemcpy(ptr, data, element_num*sizeof(float), cudaMemcpyDeviceToHost));
    }
    return arr;
}

void Tensor::_cpu()
{
    if(device == "cpu")
        return;
    else if(device == "gpu")
    {
        float* temp = new float[element_num];
        CHECK(cudaMemcpy(temp, data, element_num*sizeof(float), cudaMemcpyDeviceToHost));
        data = temp;
        manager.reset(data,"cpu"); 
    }
    else std::cout << "device error" << std::endl;
    device = "cpu";
}

void Tensor::_gpu()
{
    if(device == "gpu")
        return;
    else if(device == "cpu")
    {
        float* temp;
        CHECK(cudaMallocAsync((void**)&temp, element_num*sizeof(float), cudaStreamDefault));
        CHECK(cudaMemcpy(temp, data, element_num*sizeof(float), cudaMemcpyHostToDevice));
        data = temp;
        manager.reset(data,"gpu");
    }
    else std::cout << "device error" << std::endl;
    device = "gpu";    
}

Tensor Tensor::cpu() const
{
    assert(device == "cpu" || device == "gpu");
    Tensor cpu_tensor = this->copy();
    cpu_tensor._cpu();
    return cpu_tensor;
}

Tensor Tensor::gpu() const
{
    assert(device == "cpu" || device == "gpu");
    Tensor gpu_tensor = this->copy();
    gpu_tensor._gpu();
    return gpu_tensor;
}

Tensor Tensor::reshape(std::vector<int> _shape)
{
    assert(device == "cpu" || device == "gpu");
    Tensor res(*this);
    res._reshape(_shape); 
    return res;  
}

void Tensor::_reshape(std::vector<int> _shape)
{
    assert(device == "cpu" || device == "gpu");
    //decide which place to fill automatically
    int auto_place = -1;
    for(int i = 0 ; i < _shape.size() ; i ++)
    {
        if(_shape[i] == -1)
        {
            assert(auto_place == -1); // "reshape error: more than one -1 in shape"
            auto_place = i;
        }
    }
    //check if the shape is valid for auto filling up
    int new_elemant_num = 1;
    for(int i = 0 ; i < _shape.size() ; i ++)
    {
        new_elemant_num *= _shape[i];
    }
    if(auto_place != -1)
    {
        assert(element_num % new_elemant_num == 0); //"auto shape failed: element number not match"
        _shape[auto_place] = element_num / new_elemant_num;
    }
    else
    {
        assert(new_elemant_num == element_num);  // "reshape error: element number not match"
    }

    shape = _shape;
    ndim = shape.size();
    strides = std::vector<int>(ndim, 0);
    int stride = 1;
    for(int i = 0 ; i < ndim ; i ++)
    {
        strides[ndim-i-1] = stride;
        stride *= shape[ndim-i-1];
    }
}

Tensor Tensor::squeeze(int axis)
{
    assert(device == "cpu" || device == "gpu");
    Tensor new_tensor(*this);
    new_tensor._squeeze(axis);
    return new_tensor;
}

Tensor Tensor::unsqueeze(int axis)
{ 
    assert(device == "cpu" || device == "gpu");
    Tensor new_tensor(*this);
    new_tensor._unsqueeze(axis);
    return new_tensor;
}

void Tensor::_squeeze(int axis)
{
    assert(device == "cpu" || device == "gpu");
    assert(axis >= -ndim && axis < ndim); // "squeeze error: axis out of range"
    if(axis < 0)
        axis += ndim;
    assert(shape[axis] == 1); // "squeeze error: axis not 1"
    std::vector<int> new_shape(shape.begin(), shape.begin()+axis);
    new_shape.insert(new_shape.end(), shape.begin()+axis+1, shape.end());
    shape = new_shape;
    ndim = shape.size();
    strides = std::vector<int>(ndim, 0);
    int stride = 1;
    for(int i = 0 ; i < ndim ; i ++)
    {
        strides[ndim-i-1] = stride;
        stride *= shape[ndim-i-1];
    }
}

void Tensor::_unsqueeze(int axis)
{

    assert(device == "cpu" || device == "gpu");
    assert(axis >= -ndim-1 && axis < ndim+1); // "unsqueeze error: axis out of range"
    if(axis < 0)
        axis += ndim+1;
    std::vector<int> new_shape(shape.begin(), shape.begin()+axis);
    new_shape.push_back(1);
    new_shape.insert(new_shape.end(), shape.begin()+axis, shape.end());
    shape = new_shape;
    ndim = shape.size();
    strides = std::vector<int>(ndim, 0);
    int stride = 1;
    for(int i = 0 ; i < ndim ; i ++)
    {
        strides[ndim-i-1] = stride;
        stride *= shape[ndim-i-1];
    }
}

//return a new tensor of _shape, which is broadcasted from the original tensor and have its own memory
Tensor& Tensor::broadcast(const std::vector<int>& _shape)
{
    
    std::cout<<"broadcast not support yet"<<std::endl;
    return *this;
}

Tensor& Tensor::slice(std::vector<int> start, std::vector<int> end)
{
    std::cout << "slice not support yet" << std::endl;
    return *this;
}

Tensor& Tensor::slice(std::vector<int> start, std::vector<int> end, std::vector<int> step)
{
    std::cout << "slice not support yet" << std::endl;
    return *this;
}

float& Tensor::at(const std::vector<int>& index)
{
    assert(device == "cpu" || device == "gpu");
    if(device == "gpu")
        std::cout << "the tensor is on gpu, you should call cpu() first" << std::endl;
    int id = 0;
    for(int i = 0 ; i < ndim ; i ++)
    {
        id += index[i]*strides[i];
    }
    return data[id];
}

//deep copy of a tensor
Tensor Tensor::copy() const
{
    assert(device == "cpu" || device == "gpu");
    Tensor res(*this);
    if(device == "cpu")
    {
        res.data = new float[element_num];
        memcpy(res.data, data, element_num*sizeof(float));
    }
    else if(device == "gpu")
    {
        CHECK(cudaMallocAsync((void**)&(res.data), element_num*sizeof(float), cudaStreamDefault));
        CHECK(cudaMemcpy(res.data, data, element_num*sizeof(float), cudaMemcpyDeviceToDevice));
    }
    res.manager.reset(res.data, device);
    return res;
}   


Tensor Tensor::operator+(const Tensor& other){
    assert(device == other.device);
    Tensor res = this->copy();
    if(device == "cpu")
        add_cpu(res.data, other.data, element_num);
    else
    {
        add_gpu(res.data, other.data, element_num);
        sync_and_check_cuda_error();
    }
    return res;
}
Tensor Tensor::operator-(const Tensor& other){
    assert(device == other.device);
    Tensor res = this->copy();
    if(device == "cpu")
        dec_cpu(res.data, other.data, element_num);
    else
    {
        dec_gpu(res.data, other.data, element_num);
        sync_and_check_cuda_error();
    }
    return res;
}
Tensor Tensor::operator*(const Tensor& other){
    assert(device == other.device);
    Tensor res = this->copy();
    if(device == "cpu")
        dot_cpu(res.data, other.data, element_num);
    else
    {
        dot_gpu(res.data, other.data, element_num);
        sync_and_check_cuda_error();
    }
    return res;
}
Tensor Tensor::operator/(const Tensor& other){
    assert(device == other.device);
    Tensor res = this->copy();
    if(device == "cpu")
        div_cpu(res.data, other.data, element_num);
    else
    {
        div_gpu(res.data, other.data, element_num);
        sync_and_check_cuda_error();
    }
    return res;
}
Tensor Tensor::operator+(float other){
    Tensor res = this->copy();
    if(device == "cpu")
        add_cpu(res.data, other, element_num);
    else
    {
        add_gpu(res.data, other, element_num);
        sync_and_check_cuda_error();
    }
    return res;
}
Tensor Tensor::operator-(float other){
    Tensor res = this->copy();
    if(device == "cpu")
        dec_cpu(res.data, other, element_num);
    else
    {
        dec_gpu(res.data, other, element_num);
        sync_and_check_cuda_error();
    }  
    return res;
}
Tensor Tensor::operator*(float other){
    Tensor res = this->copy();
    if(device == "cpu")
        dot_cpu(res.data, other, element_num);
    else
    {
        dot_gpu(res.data, other, element_num);
        sync_and_check_cuda_error();
    }
    return res;
}
Tensor Tensor::operator/(float other){
    Tensor res = this->copy();
    if(device == "cpu")
        div_cpu(res.data, other, element_num);
    else
        div_gpu(res.data, other, element_num);
        sync_and_check_cuda_error();
    return res;
}
void Tensor::operator+=(const Tensor& other){
    assert(device == other.device);
    if(device == "cpu")
        add_cpu(data, other.data, element_num);
    else
        add_gpu(data, other.data, element_num);
        sync_and_check_cuda_error();
}
void Tensor::operator-=(const Tensor& other){
    assert(device == other.device);
    if(device == "cpu")
        dec_cpu(data, other.data, element_num);
    else
        dec_gpu(data, other.data, element_num);
        sync_and_check_cuda_error();
}
void Tensor::operator*=(const Tensor& other){
    assert(device == other.device);
    if(device == "cpu")
        dot_cpu(data, other.data, element_num);
    else
        dot_gpu(data, other.data, element_num);
        sync_and_check_cuda_error();
}
void Tensor::operator/=(const Tensor& other){
    assert(device == other.device);
    if(device == "cpu")
        div_cpu(data, other.data, element_num);
    else
        div_gpu(data, other.data, element_num);
        sync_and_check_cuda_error();
}
void Tensor::operator+=(float other){
    if(device == "cpu")
        add_cpu(data, other, element_num);
    else
        add_gpu(data, other, element_num);
        sync_and_check_cuda_error();
}
void Tensor::operator-=(float other){
    if(device == "cpu")
        dec_cpu(data, other, element_num);
    else
        dec_gpu(data, other, element_num);
        sync_and_check_cuda_error();
}
void Tensor::operator*=(float other){
    if(device == "cpu")
        dot_cpu(data, other, element_num);
    else
        dot_gpu(data, other, element_num);
        sync_and_check_cuda_error();
}
void Tensor::operator/=(float other){
    if(device == "cpu")
        div_cpu(data, other, element_num);
    else
        div_gpu(data, other, element_num);
        sync_and_check_cuda_error();
}

void Tensor::operator=(const Tensor& other){
    assert(device == other.device);
    assert(other.shape == shape);
    if(data == other.data)
        return;
    if(device == "cpu")
        memcpy(data, other.data, element_num*sizeof(float));
    else
        CHECK(cudaMemcpy(data, other.data, element_num*sizeof(float), cudaMemcpyDeviceToDevice));
}
void Tensor::operator=(float other){
    if(device == "cpu")
        fill_cpu(data, other, element_num);
    else
        fill_gpu(data, other, element_num);
        sync_and_check_cuda_error();
}

//[[[],[]],[[],[]]]
void print_data_recursive(std::vector<int> shape, std::vector<int> strides, float* data)
{
    std::cout << '[';
    if(shape.size() == 1)
    {
        std::cout << data[0];
        for(int i = 1 ; i < shape[0] ; i ++)
        {
            std::cout<< "," << data[i];
        }
    }
    else
    {
        for(int i = 0 ; i < shape[0] ; i ++)
        {
            print_data_recursive(std::vector<int>(shape.begin()+1, shape.end()), std::vector<int>(strides.begin()+1, strides.end()), data+i*strides[0]);
            if(i != shape[0]-1)
                std::cout << ',';
        }
    }
    std::cout << ']';
}

void Tensor::print_data() const
{
    if(device == "gpu")
    {
        Tensor temp_tensor = this->cpu();
        print_data_recursive(temp_tensor.shape, temp_tensor.strides, temp_tensor.data);
        std::cout << std::endl;
    }
    else
    {
        print_data_recursive(shape, strides, data);
        std::cout << std::endl;
    }
}

void Tensor::fill(float scalar)
{
    if(device == "cpu")
    {
        fill_cpu(data, scalar, element_num);
    }
    else if(device == "gpu")
    {
        fill_gpu(data, scalar, element_num);
        sync_and_check_cuda_error();
    }
}

std::vector<int> Tensor::get_shape() const
{
    return shape;
}

const std::vector<int>& Tensor::get_strides() const
{
    return strides;
}

int Tensor::get_ndim() const
{
    return ndim;
}

int Tensor::get_element_num() const
{
    return element_num;
}

std::string Tensor::get_device() const
{
    return device;
}

void Tensor::print_information() const
{
    std::cout << "ndim: " << ndim << std::endl;
    std::cout << "shape: ";
    for(int i = 0 ; i < ndim ; i ++)
    {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "strides: ";
    for(int i = 0 ; i < ndim ; i ++)
    {
        std::cout << strides[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "element_num: " << element_num << std::endl;
    std::cout << "device: " << device << std::endl;

    std::cout << "data_location: " << data << std::endl;
    std::cout << "refcount: " << get_refcount() << std::endl;

    print_data();
}

float* Tensor::get_data()
{
    return data;
}

const float* Tensor::read_data() const
{
    return data;
}

int Tensor::get_refcount() const
{
    return *(manager.refcount);
}

//shallow copy
void Tensor::copy_from(const Tensor& other)
{
    shape = other.shape;
    ndim = other.ndim;
    strides = other.strides;
    element_num = other.element_num;
    device = other.device;
    data = other.data;
    manager = other.manager;  //responsible for memory management
}

