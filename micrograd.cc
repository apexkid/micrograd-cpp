#include "micrograd.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <vector>

namespace apexkid {
namespace micrograd {

GradNode::GradNode(double data, std::string label,
                   std::vector<std::shared_ptr<GradNode>> children,
                   std::function<void()> backward_fn) {
  this->data = data;
  this->label = label;
  this->children = children;
  this->backward_fn = backward_fn;
}

GradNode::GradNode(double data, std::string label) {
  this->data = data;
  this->label = label;
}

void GradNode::make_scalar() { is_scalar = true; }

std::shared_ptr<GradNode> GradNode::create_gradnode(double data,
                                                    std::string label) {
  return std::make_shared<GradNode>(data, label);
}

std::shared_ptr<GradNode>
GradNode::create_gradnode(double data, std::string label,
                          std::vector<std::shared_ptr<GradNode>> children,
                          std::function<void()> backward_fn) {
  return std::make_shared<GradNode>(data, label, children, backward_fn);
}

void GradNode::backward() {
  grad = 1.0;
  auto node_stack = TopologicalSort();
  while (!node_stack.empty()) {
    auto *node = node_stack.top();
    if (node->backward_fn == nullptr) {
      node_stack.pop();
      continue;
    }
    node->backward_fn();
    node_stack.pop();
  }
}

void GradNode::print_network() {
  auto node_stack = TopologicalSort();
  while (!node_stack.empty()) {
    auto *node = node_stack.top();
    std::cout << "Label:" << node->label << " Data:" << node->data
              << " Grad:" << node->grad << std::endl;
    node_stack.pop();
  }
}

std::stack<const GradNode *> GradNode::TopologicalSort() {
  std::stack<const GradNode *> stack;
  std::set<const GradNode *> visited;

  TopologicalSortUtil(*this, stack, visited);
  return stack;
}

void GradNode::TopologicalSortUtil(const GradNode &node,
                                   std::stack<const GradNode *> &stack,
                                   std::set<const GradNode *> &visited) {
  if (visited.find(&node) != visited.end()) {
    return;
  }

  for (auto &child : node.children) {
    TopologicalSortUtil(*child, stack, visited);
  }

  stack.push(&node);
  visited.emplace(&node);
}

std::shared_ptr<GradNode> operator+(double a,
                                    const std::shared_ptr<GradNode> &b) {
  auto gradnode = GradNode::create_gradnode(a, std::to_string(a));
  gradnode->make_scalar();
  return gradnode + b;
}

std::shared_ptr<GradNode> operator+(const std::shared_ptr<GradNode> &a,
                                    double b) {
  auto gradnode = GradNode::create_gradnode(b, std::to_string(b));
  gradnode->make_scalar();
  return a + gradnode;
}

std::shared_ptr<GradNode> operator+(const std::shared_ptr<GradNode> &a,
                                    const std::shared_ptr<GradNode> &b) {
  auto output_backward = [a, b]() {
    if (!a->is_scalar) {
      a->grad += 1.0;
    }
    if (!b->is_scalar) {
      b->grad += 1.0;
    }
  };

  auto output_data = a->data + b->data;
  auto output_label = a->label + "+" + b->label;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};
  return GradNode::create_gradnode(output_data, output_label, output_children,
                                   output_backward);
}

std::shared_ptr<GradNode> operator-(double a,
                                    const std::shared_ptr<GradNode> &b) {
  auto gradnode = GradNode::create_gradnode(a, std::to_string(a));
  gradnode->make_scalar();
  return gradnode - b;
}

std::shared_ptr<GradNode> operator-(const std::shared_ptr<GradNode> &a,
                                    double b) {
  auto gradnode = GradNode::create_gradnode(b, std::to_string(b));
  gradnode->make_scalar();
  return a - gradnode;
}

std::shared_ptr<GradNode> operator-(const std::shared_ptr<GradNode> &a,
                                    const std::shared_ptr<GradNode> &b) {
  auto output_backward = [a, b]() {
    if (!a->is_scalar) {
      a->grad += 1.0;
    }
    if (!b->is_scalar) {
      b->grad -= 1.0;
    }
  };

  auto output_data = a->data - b->data;
  auto output_label = a->label + "-" + b->label;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};
  return GradNode::create_gradnode(output_data, output_label, output_children,
                                   output_backward);
}

std::shared_ptr<GradNode> operator*(double a,
                                    const std::shared_ptr<GradNode> &b) {
  auto gradnode = GradNode::create_gradnode(a, std::to_string(a));
  gradnode->make_scalar();
  return gradnode * b;
}

std::shared_ptr<GradNode> operator*(const std::shared_ptr<GradNode> &a,
                                    double b) {
  auto gradnode = GradNode::create_gradnode(b, std::to_string(b));
  gradnode->make_scalar();
  return a * gradnode;
}

std::shared_ptr<GradNode> operator*(const std::shared_ptr<GradNode> &a,
                                    const std::shared_ptr<GradNode> &b) {
  auto output_backward = [a, b]() {
    if (!a->is_scalar) {
      a->grad += b->data;
    }
    if (!b->is_scalar) {
      b->grad += a->data;
    }
  };

  auto output_data = a->data * b->data;
  auto output_label = a->label + "*" + b->label;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};
  return GradNode::create_gradnode(output_data, output_label, output_children,
                                   output_backward);
}

std::shared_ptr<GradNode> operator/(double a,
                                    const std::shared_ptr<GradNode> &b) {
  auto gradnode = GradNode::create_gradnode(a, std::to_string(a));
  gradnode->make_scalar();
  return gradnode / b;
}

std::shared_ptr<GradNode> operator/(const std::shared_ptr<GradNode> &a,
                                    double b) {
  auto gradnode = GradNode::create_gradnode(b, std::to_string(b));
  gradnode->make_scalar();
  return a / gradnode;
}

std::shared_ptr<GradNode> operator/(const std::shared_ptr<GradNode> &a,
                                    const std::shared_ptr<GradNode> &b) {
  auto output_backward = [a, b]() {
    if (!a->is_scalar) {
      a->grad += (1.0 / b->data);
    }
    if (!b->is_scalar) {
      b->grad -= (a->data / std::pow(b->data, 2));
    }
  };

  auto output_data = a->data * b->data;
  auto output_label = a->label + "*" + b->label;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};
  return GradNode::create_gradnode(output_data, output_label, output_children,
                                   output_backward);
}

} // namespace micrograd
} // namespace apexkid