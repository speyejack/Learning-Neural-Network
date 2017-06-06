
#ifndef LLEARN_TESTS_COMMON_EXPECT_INCLUDE
#define LLEARN_TESTS_COMMON_EXPECT_INCLUDE
#include <string>

void expectTrue(std::string message, bool condition);
void expectFalse(std::string message, bool condition);
void warn(std::string message);

#endif
