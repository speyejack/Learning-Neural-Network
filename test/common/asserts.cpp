#include <string>
#include <iostream>
#include <stdlib.h>
#include "asserts.h"

void assertTrue(std::string message, bool condition){
	if (!condition){
		fail(message);
	}
}

void assertFalse(std::string message, bool condition){
	assertTrue(message, !condition);
}

void warn(std::string message){
	std::cout << message << "\n";
}

void fail(std::string message){
	warn(message);
	exit(1);
}
