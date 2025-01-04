# Micrograd-cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)


A simple autograd library written in C++, inspired by [karpathy/micrograd](https://github.com/karpathy/micrograd?tab=readme-ov-file). It enables calculating gradients of mathematical equations with respect to their elements by implementing a backpropagation-like API, similar to those found in Python frameworks like PyTorch. This project was a fun, educational exercise to explore the complexities of implementing such an API in C++ as closely as possible to its Python counterparts. However, it can be used to build an actual Neural network model. Check source.


# Example usage

- Refer `micrograd_test.cc` for detailed usage.
- Supports common 4 mathematical operations `+ - * /`
- Supports calculating exponents via `pow(..)` and `log(..)`.
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
- [nn_linear_regression_demo.cc](nn_linear_regression_demo.cc) => Implements a simple linear regression over a synthetic housing data using Stochastic Gradient Descent.

Sample Run:
```
INFO: Running command line: bazel-bin/nn_linear_regression_demo
Epoch: 0 Loss: 4245.27
Epoch: 100 Loss: 49.3228
Epoch: 200 Loss: 40.2514
Epoch: 300 Loss: 34.2934
Epoch: 400 Loss: 30.0342
Epoch: 500 Loss: 26.9828
Epoch: 600 Loss: 24.796
Epoch: 700 Loss: 23.2285
Epoch: 800 Loss: 22.1045
Epoch: 900 Loss: 21.2983
Epoch: 1000 Loss: 20.7198
Epoch: 1100 Loss: 20.3044
Epoch: 1200 Loss: 20.0061
Epoch: 1300 Loss: 19.7916
Epoch: 1400 Loss: 19.6374
Epoch: 1500 Loss: 19.5263
Epoch: 1600 Loss: 19.4462
Epoch: 1700 Loss: 19.3885
Epoch: 1800 Loss: 19.3467
Epoch: 1900 Loss: 19.3165
Epoch: 2000 Loss: 19.2946
Epoch: 2100 Loss: 19.2786
Epoch: 2200 Loss: 19.267
Epoch: 2300 Loss: 19.2585
Epoch: 2400 Loss: 19.2523
Epoch: 2500 Loss: 19.2477
Epoch: 2600 Loss: 19.2443
Epoch: 2700 Loss: 19.2418
Epoch: 2800 Loss: 19.24
Epoch: 2900 Loss: 19.2386
Epoch: 3000 Loss: 19.2375
Epoch: 3100 Loss: 19.2367
Epoch: 3200 Loss: 19.2361
Epoch: 3300 Loss: 19.2356
Epoch: 3400 Loss: 19.2352
Epoch: 3500 Loss: 19.2349
Epoch: 3600 Loss: 19.2347
Epoch: 3700 Loss: 19.2345
Epoch: 3800 Loss: 19.2344
Epoch: 3900 Loss: 19.2343
Epoch: 4000 Loss: 19.2342
Epoch: 4100 Loss: 19.2341
Epoch: 4200 Loss: 19.2341
Epoch: 4300 Loss: 19.234
Epoch: 4400 Loss: 19.234
Epoch: 4500 Loss: 19.2339
Epoch: 4600 Loss: 19.2339
Epoch: 4700 Loss: 19.2339
Epoch: 4800 Loss: 19.2339
Epoch: 4900 Loss: 19.2338
Epoch: 5000 Loss: 19.2338
....
....
Epoch: 9400 Loss: 19.2338
Epoch: 9500 Loss: 19.2338
Epoch: 9600 Loss: 19.2338
Epoch: 9700 Loss: 19.2338
Epoch: 9800 Loss: 19.2338
Epoch: 9900 Loss: 19.2338
Final weights: w1=1.70961 w2=-3.04562 w3=4.07157 b=6.54784
```


- [nn_logistic_regression_demo.cc](nn_logistic_regression_demo.cc) => Implements a simple logistic regression over synthetic housing data to classify houses as expensive or cheap.

Sample Run:
```
INFO: Running command line: bazel-bin/nn_logistic_regression_demo
Epoch: 0 Loss: 11.0941
Epoch: 100 Loss: 5.86457
Epoch: 200 Loss: 4.26492
Epoch: 300 Loss: 3.41361
Epoch: 400 Loss: 2.91583
Epoch: 500 Loss: 2.59209
Epoch: 600 Loss: 2.36284
Epoch: 700 Loss: 2.18991
Epoch: 800 Loss: 2.05322
Epoch: 900 Loss: 1.94131
Epoch: 1000 Loss: 1.84717
Epoch: 1100 Loss: 1.76629
Epoch: 1200 Loss: 1.6956
Epoch: 1300 Loss: 1.63297
Epoch: 1400 Loss: 1.57685
Epoch: 1500 Loss: 1.52607
Epoch: 1600 Loss: 1.47976
Epoch: 1700 Loss: 1.43724
Epoch: 1800 Loss: 1.39795
Epoch: 1900 Loss: 1.36148
Epoch: 2000 Loss: 1.32746
Epoch: 2100 Loss: 1.2956
Epoch: 2200 Loss: 1.26566
Epoch: 2300 Loss: 1.23744
Epoch: 2400 Loss: 1.21075
Epoch: 2500 Loss: 1.18547
Epoch: 2600 Loss: 1.16145
Epoch: 2700 Loss: 1.13858
Epoch: 2800 Loss: 1.11678
Epoch: 2900 Loss: 1.09594
Epoch: 3000 Loss: 1.07601
Epoch: 3100 Loss: 1.0569
Epoch: 3200 Loss: 1.03857
Epoch: 3300 Loss: 1.02096
Epoch: 3400 Loss: 1.00402
Epoch: 3500 Loss: 0.987701
Epoch: 3600 Loss: 0.971974
Epoch: 3700 Loss: 0.956798
Epoch: 3800 Loss: 0.942143
Epoch: 3900 Loss: 0.927977
Epoch: 4000 Loss: 0.914274
Epoch: 4100 Loss: 0.901009
Epoch: 4200 Loss: 0.888158
Epoch: 4300 Loss: 0.875701
Epoch: 4400 Loss: 0.863617
Epoch: 4500 Loss: 0.851889
Epoch: 4600 Loss: 0.840499
Epoch: 4700 Loss: 0.829431
...
...
Epoch: 9100 Loss: 0.529141
Epoch: 9200 Loss: 0.524858
Epoch: 9300 Loss: 0.520645
Epoch: 9400 Loss: 0.516498
Epoch: 9500 Loss: 0.512418
Epoch: 9600 Loss: 0.508402
Epoch: 9700 Loss: 0.504449
Epoch: 9800 Loss: 0.500556
Epoch: 9900 Loss: 0.496724
Final weights: w1=2.11793 w2=-2.45353 w3=0.960559 b=-3.28885
```


# License
MIT
