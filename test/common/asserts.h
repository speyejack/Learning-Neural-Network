#ifndef LLEARN_TESTS_COMMON_ASSERT_INCLUDE
#define LLEARN_TESTS_COMMON_ASSERT_INCLUDE
#include <string>

void assertTrue(std::string message, bool condition);
void assertFalse(std::string message, bool condition);
void fail(std::string message);
void warn(std::string message);

#endif
