#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class Value {
public:
  Value(double data, std::string label, std::vector<Value *> children,
        std::function<void(std::vector<Value *>)> backward_fn) {
    this->data = data;
    this->label = label;
    this->children = children;
    this->backward_fn = backward_fn;
  }

  Value(double data, std::string label) {
    this->data = data;
    this->label = label;
  }

  static std::shared_ptr<Value> create_value(double data, std::string label) {
    return std::make_shared<Value>(data, label);
  }

  double data;
  double grad = 0.0;
  std::vector<Value *> children;
  std::string label;

  std::shared_ptr<Value> operator+(Value &rhs) {
    auto backward = [](std::vector<Value *> children) {
      for (auto &child : children) {
        child->grad += 1.0;
      }
    };
    return std::make_shared<Value>(
        Value(data + rhs.data, (this->label + "+" + rhs.label),
              std::vector<Value *>{this, &rhs}, backward));
  }

  void backward() {
    grad = 1.0;
    backward_fn(children);
  }

  void print_network() {
    std::cout << "Label: " << label << " Value: " << data << " Grad: " << grad
              << std::endl;
    for (auto &child : children) {
      child->print_network();
    }
  }

private:
  std::function<void(std::vector<Value *>)> backward_fn;
};

int main() {
  std::cout << "Hello, World!" << std::endl;
  Value a(1.0, "a");
  Value b(2.0, "b");
  Value c{3.0, "d"};
  std::shared_ptr<Value> d = a + b;
  std::shared_ptr<Value> e = *d + c;
  std::cout << c.data << std::endl;
  e->backward();
  e->print_network();
  //   std::cout << a.grad << std::endl;
  //   std::cout << b.grad << std::endl;
  //   std::cout << "c.children.size() " << c.children.size() << std::endl;
  return 0;
}