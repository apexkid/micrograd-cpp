#include "micrograd.h"
#include <iostream>

using namespace apexkid::micrograd;

int main() {
  auto a = GradNode::CreateGradnode(1.0, "a");
  auto b = GradNode::CreateGradnode(2.0, "b");
  auto c = a->pow(b);
  auto d = c + 1.0;
  d->Backward();
  d->PrintNetwork();
  return 0;
}