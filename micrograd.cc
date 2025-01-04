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
  this->data_ = data;
  this->label_ = label;
  this->children_ = children;
  this->backward_fn_ = backward_fn;
}

GradNode::GradNode(double data, std::string label) {
  this->data_ = data;
  this->label_ = label;
}

void GradNode::MakeScalar() { is_scalar_ = true; }

std::shared_ptr<GradNode> GradNode::CreateGradnode(double data,
                                                   std::string label) {
  return std::make_shared<GradNode>(data, label);
}

double GradNode::GetGrad() { return grad_; }
double GradNode::GetData() { return data_; }

std::shared_ptr<GradNode>
GradNode::CreateGradnode(double data, std::string label,
                         std::vector<std::shared_ptr<GradNode>> children,
                         std::function<void()> backward_fn) {
  return std::make_shared<GradNode>(data, label, children, backward_fn);
}

void GradNode::Backward() {
  grad_ = 1.0;
  auto node_stack = TopologicalSort();
  while (!node_stack.empty()) {
    auto *node = node_stack.top();
    if (node->backward_fn_ == nullptr) {
      node_stack.pop();
      continue;
    }
    node->backward_fn_();
    node_stack.pop();
  }
}

void GradNode::PrintNetwork() {
  auto node_stack = TopologicalSort();
  while (!node_stack.empty()) {
    auto *node = node_stack.top();
    std::cout << "Label:" << node->label_ << " Data:" << node->data_
              << " Grad:" << node->grad_ << std::endl;
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

  for (auto &child : node.children_) {
    TopologicalSortUtil(*child, stack, visited);
  }

  stack.push(&node);
  visited.emplace(&node);
}

std::shared_ptr<GradNode> operator+(double a,
                                    const std::shared_ptr<GradNode> &b) {
  auto gradnode = GradNode::CreateGradnode(a, std::to_string(a));
  gradnode->MakeScalar();
  return gradnode + b;
}

std::shared_ptr<GradNode> operator+(const std::shared_ptr<GradNode> &a,
                                    double b) {
  auto gradnode = GradNode::CreateGradnode(b, std::to_string(b));
  gradnode->MakeScalar();
  return a + gradnode;
}

std::shared_ptr<GradNode> operator+(const std::shared_ptr<GradNode> &a,
                                    const std::shared_ptr<GradNode> &b) {

  auto output_data = a->data_ + b->data_;
  auto output_label = a->label_ + "+" + b->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};

  auto result = GradNode::CreateGradnode(output_data, output_label);
  result->children_ = output_children;
  result->backward_fn_ = [a, b, result]() {
    if (!a->is_scalar_) {
      a->grad_ += result->grad_;
    }
    if (!b->is_scalar_) {
      b->grad_ += result->grad_;
    }
  };

  return result;
}

std::shared_ptr<GradNode> operator-(double a,
                                    const std::shared_ptr<GradNode> &b) {
  auto gradnode = GradNode::CreateGradnode(a, std::to_string(a));
  gradnode->MakeScalar();
  return gradnode - b;
}

std::shared_ptr<GradNode> operator-(const std::shared_ptr<GradNode> &a,
                                    double b) {
  auto gradnode = GradNode::CreateGradnode(b, std::to_string(b));
  gradnode->MakeScalar();
  return a - gradnode;
}

std::shared_ptr<GradNode> operator-(const std::shared_ptr<GradNode> &a,
                                    const std::shared_ptr<GradNode> &b) {
  auto output_data = a->data_ - b->data_;
  auto output_label = a->label_ + "-" + b->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};

  auto result = GradNode::CreateGradnode(output_data, output_label);
  result->children_ = output_children;
  result->backward_fn_ = [a, b, result]() {
    if (!a->is_scalar_) {
      a->grad_ += result->grad_;
    }
    if (!b->is_scalar_) {
      b->grad_ -= result->grad_;
    }
  };

  return result;
}

std::shared_ptr<GradNode> operator*(double a,
                                    const std::shared_ptr<GradNode> &b) {
  auto gradnode = GradNode::CreateGradnode(a, std::to_string(a));
  gradnode->MakeScalar();
  return gradnode * b;
}

std::shared_ptr<GradNode> operator*(const std::shared_ptr<GradNode> &a,
                                    double b) {
  auto gradnode = GradNode::CreateGradnode(b, std::to_string(b));
  gradnode->MakeScalar();
  return a * gradnode;
}

std::shared_ptr<GradNode> operator*(const std::shared_ptr<GradNode> &a,
                                    const std::shared_ptr<GradNode> &b) {

  auto output_data = a->data_ * b->data_;
  auto output_label = a->label_ + "*" + b->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};

  auto result = GradNode::CreateGradnode(output_data, output_label);
  result->children_ = output_children;
  result->backward_fn_ = [a, b, result]() {
    if (!a->is_scalar_) {
      a->grad_ += (result->grad_ * b->data_);
    }
    if (!b->is_scalar_) {
      b->grad_ += (result->grad_ * a->data_);
    }
  };
  return result;
}

std::shared_ptr<GradNode> operator/(double a,
                                    const std::shared_ptr<GradNode> &b) {
  auto gradnode = GradNode::CreateGradnode(a, std::to_string(a));
  gradnode->MakeScalar();
  return gradnode / b;
}

std::shared_ptr<GradNode> operator/(const std::shared_ptr<GradNode> &a,
                                    double b) {
  auto gradnode = GradNode::CreateGradnode(b, std::to_string(b));
  gradnode->MakeScalar();
  return a / gradnode;
}

std::shared_ptr<GradNode> operator/(const std::shared_ptr<GradNode> &a,
                                    const std::shared_ptr<GradNode> &b) {
  auto output_data = a->data_ / b->data_;
  auto output_label = a->label_ + "/" + b->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};

  auto result = GradNode::CreateGradnode(output_data, output_label);
  result->children_ = output_children;
  result->backward_fn_ = [a, b, result]() {
    if (!a->is_scalar_) {
      a->grad_ += (result->grad_ / b->data_);
    }
    if (!b->is_scalar_) {
      b->grad_ -= (result->grad_ * a->data_ / std::pow(b->data_, 2));
    }
  };
  return result;
}

std::shared_ptr<GradNode> pow(std::shared_ptr<GradNode> &base,
                              double exponent) {
  auto gradnode = GradNode::CreateGradnode(exponent, std::to_string(exponent));
  gradnode->MakeScalar();
  return pow(base, gradnode);
}

std::shared_ptr<GradNode> pow(std::shared_ptr<GradNode> &base,
                              std::shared_ptr<GradNode> &exponent) {
  auto output_data = std::pow(base->data_, exponent->data_);
  auto output_label = base->label_ + "^" + exponent->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{base, exponent};

  auto result = GradNode::CreateGradnode(output_data, output_label);
  result->children_ = output_children;
  result->backward_fn_ = [base, exponent, result]() {
    if (!base->is_scalar_) {
      base->grad_ += result->grad_ * exponent->data_ *
                     std::pow(base->data_, exponent->data_ - 1);
    }
    if (!exponent->is_scalar_) {
      exponent->grad_ += result->grad_ *
                         std::pow(base->data_, exponent->data_) *
                         std::log(base->data_);
    }
  };
  return result;
}

std::shared_ptr<GradNode> sigmoid(std::shared_ptr<GradNode> &x) {
  auto e = GradNode::CreateGradnode(std::exp(1), "E");
  e->MakeScalar();
  auto minus_x = -1 * x;
  auto result = 1.0 / (1.0 + pow(e, minus_x));
  return result;
}

} // namespace micrograd
} // namespace apexkid