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
  auto output_backward = [a, b]() {
    if (!a->is_scalar_) {
      a->grad_ += 1.0;
    }
    if (!b->is_scalar_) {
      b->grad_ += 1.0;
    }
  };

  auto output_data = a->data_ + b->data_;
  auto output_label = a->label_ + "+" + b->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};
  return GradNode::CreateGradnode(output_data, output_label, output_children,
                                  output_backward);
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
  auto output_backward = [a, b]() {
    if (!a->is_scalar_) {
      a->grad_ += 1.0;
    }
    if (!b->is_scalar_) {
      b->grad_ -= 1.0;
    }
  };

  auto output_data = a->data_ - b->data_;
  auto output_label = a->label_ + "-" + b->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};
  return GradNode::CreateGradnode(output_data, output_label, output_children,
                                  output_backward);
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
  auto output_backward = [a, b]() {
    if (!a->is_scalar_) {
      a->grad_ += b->data_;
    }
    if (!b->is_scalar_) {
      b->grad_ += a->data_;
    }
  };

  auto output_data = a->data_ * b->data_;
  auto output_label = a->label_ + "*" + b->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};
  return GradNode::CreateGradnode(output_data, output_label, output_children,
                                  output_backward);
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
  auto output_backward = [a, b]() {
    if (!a->is_scalar_) {
      a->grad_ += (1.0 / b->data_);
    }
    if (!b->is_scalar_) {
      b->grad_ -= (a->data_ / std::pow(b->data_, 2));
    }
  };

  auto output_data = a->data_ * b->data_;
  auto output_label = a->label_ + "*" + b->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};
  return GradNode::CreateGradnode(output_data, output_label, output_children,
                                  output_backward);
}

std::shared_ptr<GradNode> GradNode::pow(double p) {
  auto gradnode = GradNode::CreateGradnode(p, std::to_string(p));
  gradnode->MakeScalar();
  return GradNode::pow(gradnode);
}

std::shared_ptr<GradNode> GradNode::pow(std::shared_ptr<GradNode> &p) {
  auto output_backward = [p, this]() {
    if (!p->is_scalar_) {
      p->grad_ += std::pow(this->data_, p->data_) * std::log(this->data_);
    }
    if (!this->is_scalar_) {
      this->grad_ += this->data_ * std::pow(this->data_, p->data_ - 1);
    }
  };

  auto output_data = std::pow(this->data_, p->data_);
  auto output_label = this->label_ + "^" + p->label_;
  auto output_children = std::vector<std::shared_ptr<GradNode>>{
      std::make_shared<GradNode>(*this), p};
  return GradNode::CreateGradnode(output_data, output_label, output_children,
                                  output_backward);
}

} // namespace micrograd
} // namespace apexkid