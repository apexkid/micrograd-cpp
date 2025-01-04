# Micrograd-cpp

A simple autograd library written entirely in C++ inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd?tab=readme-ov-file). Enables calculating gradients of any mathematical equation with respect to its elements by implementing
backpropagation like API that is commonly used by Python frameworks like Pytorch. 

The goal was to explore the complexities of C++ in trying to implement a backpropoagation API which is as close to Python frameworks as possible. Purely for fun and educational purpose. I educated myself at least :)

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
