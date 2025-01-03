#include "micrograd.h"

using namespace apexkid::micrograd;

int main() {

  auto a = GradNode::create_gradnode(1.0, "a");
  auto b = GradNode::create_gradnode(2.0, "b");
  auto c = a - b;
  auto d = c + 1.0;
  d->backward();
  d->print_network();
  return 0;
}