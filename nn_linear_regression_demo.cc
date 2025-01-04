#include "micrograd.h"
#include <iostream>
#include <vector>
using namespace apexkid::micrograd;

// Implementing a single neuron linear regression model using micrograd.
// The model is trained to predict the price of a house given the number of
// bedrooms, age of the house, and lot size in acres.
//
// @author apexkid
int main() {
  // Input features
  // Mocked using: y = 2*x1 - 3*x2 + 4*x3 + 5
  std::vector<double> x1 = {4, 2, 3, 1, 2, 8, 1, 9, 6, 1}; // Num of bedrooms
  std::vector<double> x2 = {3, 1, 4, 4, 2,
                            1, 2, 3, 2, 2}; // Age of house in years
  std::vector<double> x3 = {7, 7, 9, 3, 1, 6, 3, 5, 7, 5}; // Lot size in acres
  // Output values. Actual: {32, 34, 35, 7,  7, 42, 13, 34, 39, 21}
  std::vector<double> y = {33,   34, 35, 8.2, 7,
                           41.4, 13, 33, 39,  26}; // Price in 10000s of dollars

  // Initialize weights randomly between -1 and 1.
  auto w1 = GradNode::CreateGradnode(0.1, "w1");
  auto w2 = GradNode::CreateGradnode(0.7, "w2");
  auto w3 = GradNode::CreateGradnode(-0.4, "w3");
  auto b = GradNode::CreateGradnode(0.0, "b");

  // Learning rate
  double lr = 0.001;

  // Training loop
  for (int epoch = 0; epoch < 10000; epoch++) {
    std::shared_ptr<GradNode> loss;
    double cumulative_loss = 0;
    for (int i = 0; i < x1.size(); i++) {
      // Forward pass
      auto pred = w1 * x1[i] + w2 * x2[i] + w3 * x3[i] + b;
      auto diff = pred - y[i];
      loss = pow(diff, 2);
      cumulative_loss += loss->GetData();

      // Backward pass
      // This is an example of Stochastic Gradient Descent (SGD) as it is
      // running on each training example.
      loss->Backward();

      // Update weights
      w1 = GradNode::CreateGradnode(w1->GetData() - lr * w1->GetGrad(), "w1");
      w2 = GradNode::CreateGradnode(w2->GetData() - lr * w2->GetGrad(), "w2");
      w3 = GradNode::CreateGradnode(w3->GetData() - lr * w3->GetGrad(), "w3");
      b = GradNode::CreateGradnode(b->GetData() - lr * b->GetGrad(), "b");
    }
    if (epoch % 100 == 0) {
      std::cout << "Epoch: " << epoch << " Loss: " << cumulative_loss
                << std::endl;
    }
  }
  std::cout << "Final weights: w1=" << w1->GetData() << " w2=" << w2->GetData()
            << " w3=" << w3->GetData() << " b=" << b->GetData() << std::endl;
  return 0;
}