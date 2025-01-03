#ifndef MICROGRAD_H
#define MICROGRAD_H

#include <functional>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <vector>

namespace apexkid {
namespace micrograd {

class GradNode {
public:
  GradNode(double data, std::string label,
           std::vector<std::shared_ptr<GradNode>> children,
           std::function<void()> backward_fn);

  GradNode(double data, std::string label);

  void make_scalar();
  void backward();
  void print_network();

  static std::shared_ptr<GradNode> create_gradnode(double data,
                                                   std::string label);

  static std::shared_ptr<GradNode>
  create_gradnode(double data, std::string label,
                  std::vector<std::shared_ptr<GradNode>> children,
                  std::function<void()> backward_fn);

  // Addition
  friend std::shared_ptr<GradNode>
  operator+(double a, const std::shared_ptr<GradNode> &b);

  friend std::shared_ptr<GradNode> operator+(const std::shared_ptr<GradNode> &a,
                                             double b);

  friend std::shared_ptr<GradNode>
  operator+(const std::shared_ptr<GradNode> &a,
            const std::shared_ptr<GradNode> &b);

  // Subtraction
  friend std::shared_ptr<GradNode>
  operator-(double a, const std::shared_ptr<GradNode> &b);

  friend std::shared_ptr<GradNode> operator-(const std::shared_ptr<GradNode> &a,
                                             double b);

  friend std::shared_ptr<GradNode>
  operator-(const std::shared_ptr<GradNode> &a,
            const std::shared_ptr<GradNode> &b);

  // Multiplication
  friend std::shared_ptr<GradNode>
  operator*(double a, const std::shared_ptr<GradNode> &b);

  friend std::shared_ptr<GradNode> operator*(const std::shared_ptr<GradNode> &a,
                                             double b);

  friend std::shared_ptr<GradNode>
  operator*(const std::shared_ptr<GradNode> &a,
            const std::shared_ptr<GradNode> &b);

  // Division
  friend std::shared_ptr<GradNode>
  operator/(double a, const std::shared_ptr<GradNode> &b);

  friend std::shared_ptr<GradNode> operator/(const std::shared_ptr<GradNode> &a,
                                             double b);

  friend std::shared_ptr<GradNode>
  operator/(const std::shared_ptr<GradNode> &a,
            const std::shared_ptr<GradNode> &b);

private:
  std::stack<const GradNode *> TopologicalSort();

  void TopologicalSortUtil(const GradNode &node,
                           std::stack<const GradNode *> &stack,
                           std::set<const GradNode *> &visited);

  std::function<void()> backward_fn;

  double data;
  double grad = 0.0;
  std::vector<std::shared_ptr<GradNode>> children;
  std::string label;
  bool is_scalar = false;
};

} // namespace micrograd
} // namespace apexkid

#endif // MICROGRAD_H