#include "micrograd.h"
#include "gtest/gtest.h"

namespace apexkid {
namespace micrograd {
namespace {

TEST(MicrogradTest, Sum) {
  auto a = GradNode::CreateGradnode(1.0, "a");
  auto b = GradNode::CreateGradnode(1.0, "b");

  auto z = a + b;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 1.0);
  EXPECT_EQ(b->GetGrad(), 1.0);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 2.0);
}

TEST(MicrogradTest, Minus) {
  auto a = GradNode::CreateGradnode(1.0, "a");
  auto b = GradNode::CreateGradnode(1.0, "b");

  auto z = a - b;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 1.0);
  EXPECT_EQ(b->GetGrad(), -1.0);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 0);
}

TEST(MicrogradTest, Multiply) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto b = GradNode::CreateGradnode(4.0, "b");

  auto z = a * b;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 4);
  EXPECT_EQ(b->GetGrad(), 2);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 8.0);
}

TEST(MicrogradTest, Divide) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto b = GradNode::CreateGradnode(4.0, "b");

  auto z = a / b;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 0.25);
  EXPECT_EQ(b->GetGrad(), -(2.0 / 16));
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 0.5);
}

TEST(Micrograd, Power) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto b = GradNode::CreateGradnode(3.0, "b");

  auto z = pow(a, b);
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 12.0);
  EXPECT_EQ(b->GetGrad(), 8.0 * std::log(2.0));
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 8.0);
}

TEST(Micrograd, ScalarSum) {
  auto a = GradNode::CreateGradnode(1.0, "a");
  auto z = 1.0 + a;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 1.0);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 2.0);
}

TEST(Micrograd, ScalarMinus) {
  auto a = GradNode::CreateGradnode(1.0, "a");
  auto z = 1.0 - a;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), -1.0);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 0.0);
}

TEST(Micrograd, ScalarMultiply) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto z = 2.0 * a;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 2.0);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 4.0);
}

TEST(Micrograd, ScalarDivide) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto z = a / 2.0;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 0.5);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 1.0);
}

TEST(Micrograd, ScalarPower) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto z = pow(a, 3.0);
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 12.0);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 8.0);
}

// Z = AB + B + C
TEST(Micrograd, ChainedEquation1) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto b = GradNode::CreateGradnode(3.0, "b");
  auto c = GradNode::CreateGradnode(4.0, "c");

  auto z = (a * b) + b + c;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 3.0);
  EXPECT_EQ(b->GetGrad(), 3.0);
  EXPECT_EQ(c->GetGrad(), 1.0);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 13.0);
}

// Z = A^2 + AB + C/B + D
TEST(Micrograd, ChainedEquation2) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto b = GradNode::CreateGradnode(3.0, "b");
  auto c = GradNode::CreateGradnode(4.0, "c");
  auto d = GradNode::CreateGradnode(5.0, "d");

  auto z = d + pow(a, 2.0) + (a * b) + (c / b);
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 7.0);
  EXPECT_EQ(b->GetGrad(), 2.0 - (4.0 / 9.0));
  EXPECT_EQ(c->GetGrad(), 1.0 / 3.0);
  EXPECT_EQ(d->GetGrad(), 1.0);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 15 + 4.0 / 3);
}

// Z = ((A^2 * B) + A) / C
TEST(Micrograd, ChainedEquation3) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto b = GradNode::CreateGradnode(4.0, "b");
  auto c = GradNode::CreateGradnode(8.0, "c");

  auto z = ((pow(a, 2.0) * b) + a) / c;
  z->Backward();

  EXPECT_EQ(a->GetGrad(), 17.0 / 8.0);
  EXPECT_EQ(b->GetGrad(), 0.5);
  EXPECT_EQ(c->GetGrad(), -18.0 / 64.0);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), 18.0 / 8.0);
}

// Z = (A + B)/(A - B)
TEST(Micrograd, ChainedEquation4) {
  auto a = GradNode::CreateGradnode(2.0, "a");
  auto b = GradNode::CreateGradnode(4.0, "b");

  auto z = (a + b) / (a - b);
  z->Backward();

  EXPECT_EQ(a->GetGrad(), -2);
  EXPECT_EQ(b->GetGrad(), 1);
  EXPECT_EQ(z->GetGrad(), 1.0);
  EXPECT_EQ(z->GetData(), -3.0);
}

} // namespace
} // namespace micrograd
} // namespace apexkid
