#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <vector>

class GradNode {
public:
  GradNode(
      double data, std::string label,
      std::vector<std::shared_ptr<GradNode>> children,
      std::function<void(std::vector<std::shared_ptr<GradNode>>)> backward_fn) {
    this->data = data;
    this->label = label;
    this->children = children;
    this->backward_fn = backward_fn;
  }

  GradNode(double data, std::string label) {
    this->data = data;
    this->label = label;
  }

  void make_scalar() { is_scalar = true; }

  static std::shared_ptr<GradNode> create_grad_node(double data,
                                                    std::string label) {
    return std::make_shared<GradNode>(data, label);
  }

  static std::shared_ptr<GradNode> create_grad_node(
      double data, std::string label,
      std::vector<std::shared_ptr<GradNode>> children,
      std::function<void(std::vector<std::shared_ptr<GradNode>>)> backward_fn) {
    return std::make_shared<GradNode>(data, label, children, backward_fn);
  }

  friend std::shared_ptr<GradNode>
  operator+(double a, const std::shared_ptr<GradNode> &b) {
    auto gradnode = create_grad_node(a, std::to_string(a));
    gradnode->make_scalar();
    return gradnode + b;
  }

  friend std::shared_ptr<GradNode> operator+(const std::shared_ptr<GradNode> &a,
                                             double b) {
    auto gradnode = create_grad_node(b, std::to_string(b));
    gradnode->make_scalar();
    return a + gradnode;
  }

  friend std::shared_ptr<GradNode>
  operator+(const std::shared_ptr<GradNode> &a,
            const std::shared_ptr<GradNode> &b) {
    auto output_backward = [](std::vector<std::shared_ptr<GradNode>> children) {
      for (auto &child : children) {
        if (!child->is_scalar) {
          child->grad += 1.0;
        }
      }
    };

    auto output_data = a->data + b->data;
    auto output_label = a->label + "+" + b->label;
    auto output_children = std::vector<std::shared_ptr<GradNode>>{a, b};
    return create_grad_node(output_data, output_label, output_children,
                            output_backward);
  }

  void backward() {
    grad = 1.0;
    auto node_stack = TopologicalSort();
    while (!node_stack.empty()) {
      auto *node = node_stack.top();
      if (node->backward_fn == nullptr) {
        node_stack.pop();
        continue;
      }
      node->backward_fn(node->children);
      node_stack.pop();
    }
  }

  void print_network() {
    auto node_stack = TopologicalSort();
    while (!node_stack.empty()) {
      auto *node = node_stack.top();
      std::cout << "Label:" << node->label << " Data:" << node->data
                << " Grad:" << node->grad << std::endl;
      node_stack.pop();
    }
  }

private:
  std::stack<const GradNode *> TopologicalSort() {
    std::stack<const GradNode *> stack;
    std::set<const GradNode *> visited;

    TopologicalSortUtil(*this, stack, visited);
    return stack;
  }

  void TopologicalSortUtil(const GradNode &node,
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

  std::function<void(std::vector<std::shared_ptr<GradNode>>)> backward_fn;

  double data;
  double grad = 0.0;
  std::vector<std::shared_ptr<GradNode>> children;
  std::string label;
  bool is_scalar = false;
};

int main() {
  std::cout << "Hello, World!" << std::endl;
  auto a = GradNode::create_grad_node(1.0, "a");
  auto b = GradNode::create_grad_node(2.0, "b");
  auto c = GradNode::create_grad_node(3.0, "c");
  std::shared_ptr<GradNode> d = a + b;
  std::shared_ptr<GradNode> e = 1 + a + d + c;
  e->backward();
  e->print_network();
  return 0;
}