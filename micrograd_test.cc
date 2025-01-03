#include "micrograd.h"
#include "gtest/gtest.h"

namespace apexkid {
namespace micrograd {
namespace {

TEST(MicrogradTest, Sum) {
  auto a = GradNode::CreateGradnode(1.0, "a");
  auto b = GradNode::CreateGradnode(1.0, "b");

  auto c = a + b;
  c->Backward();

  EXPECT_EQ(a->GetGrad(), 1.0);
  EXPECT_EQ(b->GetGrad(), 1.0);
  EXPECT_EQ(c->GetGrad(), 1.0);
}

TEST(MicrogradTest, Minus) {
  auto a = GradNode::CreateGradnode(1.0, "a");
  auto b = GradNode::CreateGradnode(1.0, "b");

  auto c = a - b;
  c->Backward();

  EXPECT_EQ(a->GetGrad(), 1.0);
  EXPECT_EQ(b->GetGrad(), -1.0);
  EXPECT_EQ(c->GetGrad(), 1.0);
}

TEST(MicrogradTest, Multiply) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto b = GradNode::CreateGradnode(4.0, "b");

  auto c = a * b;
  c->Backward();

  EXPECT_EQ(a->GetGrad(), 4);
  EXPECT_EQ(b->GetGrad(), 2);
  EXPECT_EQ(c->GetGrad(), 1.0);
}

} // namespace
} // namespace micrograd
} // namespace apexkid
