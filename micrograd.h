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

/**
 * @class GradNode
 * @brief Represents a node in a computational graph for automatic
 * differentiation.
 *
 * This class encapsulates a value (data), its gradient, and a backward function
 * for calculating gradients in a computational graph. GradNode objects can be
 * linked together to form a graph that supports forward and backward passes.
 */
class GradNode {
public:
  /**
   * @brief Constructs a GradNode with data, label, children, and a backward
   * function.
   * @param data The value of the node.
   * @param label A label to identify the node.
   * @param children A vector of child nodes.
   * @param backward_fn A function that computes the gradient for the children
   * of this node.
   */
  GradNode(double data, std::string label,
           std::vector<std::shared_ptr<GradNode>> children,
           std::function<void()> backward_fn);

  /**
   * @brief Constructs a GradNode with data and label.
   * @param data The value of the node.
   * @param label A label to identify the node.
   */
  GradNode(double data, std::string label);

  /**
   * @brief Converts the node to represent a scalar value.
   */
  void MakeScalar();

  /**
   * @brief Performs a backward pass to compute gradients.
   *
   * This method computes the gradients for all nodes in the graph starting
   * from the current node.
   */
  void Backward();

  /**
   * @brief Prints the structure of the computational graph.
   */
  void PrintNetwork();

  /**
   * @brief Gets the gradient value of the node.
   * @return The gradient value.
   */
  double GetGrad();

  /**
   * @brief Gets the data value of the node.
   * @return The data value.
   */
  double GetData();

  /**
   * @brief Creates a GradNode with data and label.
   * @param data The value of the node.
   * @param label A label to identify the node.
   * @return A shared pointer to the created GradNode.
   */
  static std::shared_ptr<GradNode> CreateGradnode(double data,
                                                  std::string label);

  /**
   * @brief Creates a GradNode with data, label, children, and a backward
   * function.
   * @param data The value of the node.
   * @param label A label to identify the node.
   * @param children A vector of child nodes.
   * @param backward_fn A function that computes the gradient for the children
   * of this node.
   * @return A shared pointer to the created GradNode.
   */
  static std::shared_ptr<GradNode>
  CreateGradnode(double data, std::string label,
                 std::vector<std::shared_ptr<GradNode>> children,
                 std::function<void()> backward_fn);

  // Overloaded operators for arithmetic operations

  /// Addition
  friend std::shared_ptr<GradNode>
  operator+(double a, const std::shared_ptr<GradNode> &b);

  friend std::shared_ptr<GradNode> operator+(const std::shared_ptr<GradNode> &a,
                                             double b);

  friend std::shared_ptr<GradNode>
  operator+(const std::shared_ptr<GradNode> &a,
            const std::shared_ptr<GradNode> &b);

  /// Subtraction
  friend std::shared_ptr<GradNode>
  operator-(double a, const std::shared_ptr<GradNode> &b);

  friend std::shared_ptr<GradNode> operator-(const std::shared_ptr<GradNode> &a,
                                             double b);

  friend std::shared_ptr<GradNode>
  operator-(const std::shared_ptr<GradNode> &a,
            const std::shared_ptr<GradNode> &b);

  /// Multiplication
  friend std::shared_ptr<GradNode>
  operator*(double a, const std::shared_ptr<GradNode> &b);

  friend std::shared_ptr<GradNode> operator*(const std::shared_ptr<GradNode> &a,
                                             double b);

  friend std::shared_ptr<GradNode>
  operator*(const std::shared_ptr<GradNode> &a,
            const std::shared_ptr<GradNode> &b);

  /// Division
  friend std::shared_ptr<GradNode>
  operator/(double a, const std::shared_ptr<GradNode> &b);

  friend std::shared_ptr<GradNode> operator/(const std::shared_ptr<GradNode> &a,
                                             double b);

  friend std::shared_ptr<GradNode>
  operator/(const std::shared_ptr<GradNode> &a,
            const std::shared_ptr<GradNode> &b);

  /// Power
  friend std::shared_ptr<GradNode> pow(std::shared_ptr<GradNode> &base,
                                       double exponent);
  friend std::shared_ptr<GradNode> pow(std::shared_ptr<GradNode> &base,
                                       std::shared_ptr<GradNode> &exponent);

  /// Sigmoid
  friend std::shared_ptr<GradNode> sigmoid(std::shared_ptr<GradNode> &x);

private:
  /**
   * @brief Performs a topological sort of the computational graph.
   * @return A stack of pointers to GradNode objects in topological order.
   */
  std::stack<const GradNode *> TopologicalSort();

  /**
   * @brief Utility function for topological sort.
   * @param node The current node.
   * @param stack The stack to store the topological order.
   * @param visited A set of visited nodes.
   */
  void TopologicalSortUtil(const GradNode &node,
                           std::stack<const GradNode *> &stack,
                           std::set<const GradNode *> &visited);

  /// Private members
  std::vector<std::shared_ptr<GradNode>> children_; // Child nodes.
  std::function<void()> backward_fn_; // Backward function to compute gradients.
  double data_;                       // The value of the node.
  double grad_ = 0.0;                 // The gradient of the node.
  std::string label_;                 // The label of the node.
  bool is_scalar_ = false; // Indicates if the node represents a scalar value.
};

} // namespace micrograd
} // namespace apexkid

#endif // MICROGRAD_H
