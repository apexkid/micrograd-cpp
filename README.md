# Micrograd-cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


A simple autograd library written in C++, inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd?tab=readme-ov-file). It enables calculating gradients of mathematical equations with respect to their elements by implementing a backpropagation-like API, similar to those found in Python frameworks like PyTorch. This project was a fun, educational exercise to explore the complexities of implementing such an API in C++ as closely as possible to its Python counterparts. However, it can be used to build an actual Neural network model. Check source.


# Example usage

- Refer `micrograd_test.cc` for detailed usage.
- Supports common 4 mathematical operations `+ - * /`
- Supports calculating exponents via `pow(..)` and `exp(..)`.
- Activation functions supported -> `sigmoid, tanh, relu`. Straighforward to implement a new one.


```
// Z = ((pow(A, 2) * B) + A) / C
auto a = GradNode::CreateGradnode(2.0, "a");
auto b = GradNode::CreateGradnode(4.0, "b");
auto c = GradNode::CreateGradnode(8.0, "c");

auto z = ((pow(a, 2.0) * b) + a) / c;
auto y = sigmoid(z)
auto t = tanh(z)
auto r = relu(z)

// Calculate gradients
y->Backward();
y->PrintNetwork(); 

std::cout << a->GetGrad() << std::endl;
std::cout << y->GetData() << std::endl;
```

# Training a Neural Network

This library can be used to build a neural network as illustated in:
- nn_regression_demo => Implements a simple regression over a synthetic housing data using Stochastic Gradient Descent.

# License
MIT
