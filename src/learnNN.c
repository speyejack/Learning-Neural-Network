#include <stdio.h>
#include "matrix.h"
#include <vector>
#include "textTrainer.h"
#include "network.h"
#include <iostream>

int main(){
	std::vector<int> layers = {128, 128, 128};
	Network net(layers, {1432,114361, 19291992, 192032});
	TextTrainer t(&net, 100, 0.1, "input/dream.txt");
	for( int i = 0; i < 1000; i ++){
		t.train();
	}
	char c = -1;
	for (int i = 0; i < 20; i++){
		c = t.sample(c);
		std::cout << c;
	}
		
}
