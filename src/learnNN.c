#include <stdio.h>
#include "matrix.h"
#include <vector>
#include "textTrainer.h"
#include "network.h"
#include <iostream>
#include <ctime>
#include <string>

int main(){
	std::vector<int> layers = {128, 128};
	unsigned long seed = time(0);
	std::cout << "Seed: " << seed << std::endl;
	Network net(layers, {seed});
	
	TextTrainer t(&net, 100, 0.2, "inputs/dream.txt");
	for( int i = 0; i < 100000; i ++){
		t.train();
		std::string str = t.sample_string(-1, 20);
		std::cout << str << std::endl;
	}
	
   
}
