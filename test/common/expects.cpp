#include <string>
#include <iostream>
#include <stdlib.h>
#include "expects.h"

void expectTrue(std::string message, bool condition){
	if (!condition){
		warn(message);
	}
}

void expectFalse(std::string message, bool condition){
	expectTrue(message, !condition);
}

void warn(std::string message){
	std::cout << message << "\n";
}
