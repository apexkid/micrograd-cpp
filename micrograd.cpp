#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <vector>

class Value {
public:
  Value(double data, std::string label,
        std::vector<std::shared_ptr<Value>> children,
        std::function<void(std::vector<std::shared_ptr<Value>>)> backward_fn) {
    this->data = data;
    this->label = label;
    this->children = children;
    this->backward_fn = backward_fn;
  }

  Value(double data, std::string label) {
    this->data = data;
    this->label = label;
  }

  void make_scalar() { is_scalar = true; }

  static std::shared_ptr<Value> create_value(double data, std::string label) {
    return std::make_shared<Value>(data, label);
  }

  static std::shared_ptr<Value> create_value(
      double data, std::string label,
      std::vector<std::shared_ptr<Value>> children,
      std::function<void(std::vector<std::shared_ptr<Value>>)> backward_fn) {
    return std::make_shared<Value>(data, label, children, backward_fn);
  }

  friend std::shared_ptr<Value> operator+(double a,
                                          const std::shared_ptr<Value> &b) {
    auto value = create_value(a, std::to_string(a));
    value->make_scalar();
    return value + b;
  }

  friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a,
                                          double b) {
    auto value = create_value(b, std::to_string(b));
    value->make_scalar();
    return a + value;
  }

  friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a,
                                          const std::shared_ptr<Value> &b) {
    auto output_backward = [](std::vector<std::shared_ptr<Value>> children) {
      for (auto &child : children) {
        if (!child->is_scalar) {
          child->grad += 1.0;
        }
      }
    };

    auto output_data = a->data + b->data;
    auto output_label = a->label + "+" + b->label;
    auto output_children = std::vector<std::shared_ptr<Value>>{a, b};
    return create_value(output_data, output_label, output_children,
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
  std::stack<const Value *> TopologicalSort() {
    std::stack<const Value *> stack;
    std::set<const Value *> visited;

    TopologicalSortUtil(*this, stack, visited);
    return stack;
  }

  void TopologicalSortUtil(const Value &node, std::stack<const Value *> &stack,
                           std::set<const Value *> &visited) {
    if (visited.find(&node) != visited.end()) {
      return;
    }

    for (auto &child : node.children) {
      TopologicalSortUtil(*child, stack, visited);
    }

    stack.push(&node);
    visited.emplace(&node);
  }

  std::function<void(std::vector<std::shared_ptr<Value>>)> backward_fn;

  double data;
  double grad = 0.0;
  std::vector<std::shared_ptr<Value>> children;
  std::string label;
  bool is_scalar = false;
};

int main() {
  std::cout << "Hello, World!" << std::endl;
  auto a = Value::create_value(1.0, "a");
  auto b = Value::create_value(2.0, "b");
  auto c = Value::create_value(3.0, "c");
  std::shared_ptr<Value> d = a + b;
  std::shared_ptr<Value> e = 1 + a + d + c;
  e->backward();
  e->print_network();
  return 0;
}