#include "micrograd.h"
#include "gtest/gtest.h"

namespace apexkid {
namespace micrograd {
namespace {

TEST(MicrogradTest, Summation) {
  auto a = GradNode::create_gradnode(1.0, "a");
  auto b = GradNode::create_gradnode(1.0, "b");

  auto c = a + b;
  c->backward();

  EXPECT_EQ(a->grad, 1.0);
  EXPECT_EQ(b->grad, 1.0);
  EXPECT_EQ(c->grad, 1.0);
}

} // namespace
} // namespace micrograd
} // namespace apexkid