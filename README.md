# Pitorch
The first AI framewrok designed by myself, which is the final project of class "Programming in Artificial Intelligence"

## Code structure
The code is orgnized as follows:
```bash
├── data
│   └── MNIST
├── lib                     # C++ and Cuda code of pitorch
│   ├── cpu_func.h          # functions working on cpu
│   ├── cpu_func.cc          
│   ├── gpu_func.h          # functions working on gpu
│   ├── gpu_func.cu          
│   ├── Tensor.h            # Tensor class
│   ├── Tensor.cu          
│   ├── tensor_func.h       # wrapped cpu/gpu functions for Tensor
│   ├── tensor_func.cu          
│   ├── utils.h             # utils for error detection
│   ├── pybind.cpp          
│   └── setup.py            # install the lib
│
├── pitorch                    # Python code of pitorch
│   ├── __init__.py
│   ├── basic_operator.py      # basic elements for Computinng Graph
│   ├── Pisor.py               # difination of Pisor and Operators
│   ├── autodiff.py            # gradient back propagation
│   ├── Unittest.py            # validate C++/Cuda operators
│   ├── test_forward.py        
│   ├── test_backward.py       
│   ├── train_fc_net.py        # train a fully connected network
│   ├── train_conv_net.py      # train a convolution network            
│   └── utils.py               # some util function (not finished yet)
└── README.md
```


## Usage

### Installation
```bash
conda create -n pitorch python=3.9
conda activate pitorch
# install pytorch according to your cuda version.
# (We just use torch.utils.cpp_extension to conveniently bind cuda code to python, and not utilize torch to conduct any operation)
cd lib
python setup.py develop
```

### Running demo
You can start training on MNIST as follows.
```bash
cd pitorch
# train convolution network
python train_conv_net.py
# train fully connected network
python train_fc_net.py
```
You can tweak hyper-parameters as well as change device on the bottom of training script. 

Here is an example of training log:
<div align=center><img src="./assets/train_on_minist.png" alt="Image" width="80%"></div>
