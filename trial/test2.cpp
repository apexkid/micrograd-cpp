#include <functional>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

// Define a Node structure for the computational graph
struct Node {
  double value;                               // The value of the node
  double grad;                                // The gradient of the node
  std::vector<std::shared_ptr<Node>> parents; // Parent nodes in the graph
  std::function<void()> backward_fn; // Backward function to compute gradients

  Node(double val) : value(val), grad(0.0) {}

  void backward() {
    grad = 1.0; // Assume this is the starting node (loss node)
    backward_fn();
  }

  // Overload + operator
  friend std::shared_ptr<Node> operator+(const std::shared_ptr<Node> &a,
                                         const std::shared_ptr<Node> &b) {
    auto result = std::make_shared<Node>(a->value + b->value);
    result->parents = {a, b};
    result->backward_fn = [result, a, b]() {
      a->grad += result->grad;
      b->grad += result->grad;
    };
    return result;
  }

  // Overload * operator
  friend std::shared_ptr<Node> operator*(const std::shared_ptr<Node> &a,
                                         const std::shared_ptr<Node> &b) {
    auto result = std::make_shared<Node>(a->value * b->value);
    result->parents = {a, b};
    result->backward_fn = [result, a, b]() {
      a->grad += b->value * result->grad;
      b->grad += a->value * result->grad;
    };
    return result;
  }

  // Overload - operator
  friend std::shared_ptr<Node> operator-(const std::shared_ptr<Node> &a,
                                         const std::shared_ptr<Node> &b) {
    auto result = std::make_shared<Node>(a->value - b->value);
    result->parents = {a, b};
    result->backward_fn = [result, a, b]() {
      a->grad += result->grad;
      b->grad -= result->grad;
    };
    return result;
  }

  // Overload / operator
  friend std::shared_ptr<Node> operator/(const std::shared_ptr<Node> &a,
                                         const std::shared_ptr<Node> &b) {
    auto result = std::make_shared<Node>(a->value / b->value);
    result->parents = {a, b};
    result->backward_fn = [result, a, b]() {
      a->grad += (1 / b->value) * result->grad;
      b->grad -= (a->value / (b->value * b->value)) * result->grad;
    };
    return result;
  }
};

// Helper function to create a shared pointer for a Node
std::shared_ptr<Node> create_node(double value) {
  return std::make_shared<Node>(value);
}

// Perform backward propagation through the graph
void backward(const std::shared_ptr<Node> &node) {
  node->grad = 1.0;
  std::vector<std::shared_ptr<Node>> stack = {node};

  while (!stack.empty()) {
    auto current = stack.back();
    stack.pop_back();

    if (current->backward_fn) {
      current->backward_fn();
    }

    for (const auto &parent : current->parents) {
      stack.push_back(parent);
    }
  }
}

// Example usage
int main() {
  auto x = create_node(2.0);
  auto y = create_node(3.0);

  // z = x * y + y^2
  auto z = x * y + y;

  std::cout << "Forward value of z: " << z->value << std::endl;

  // Perform backward pass
  backward(z);

  std::cout << "Gradient of x: " << x->grad << std::endl;
  std::cout << "Gradient of y: " << y->grad << std::endl;

  return 0;
}
