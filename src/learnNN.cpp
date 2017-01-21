#include <stdio.h>
#include "matrix.h"
#include <vector>
#include "textTrainer.h"
#include "network.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>

int main(){
	std::vector<int> layers = {128, 128};
	unsigned long seed = time(0);
	std::cout << "Seed: " << seed << std::endl;
	Network net(layers, {seed});
	
	TextTrainer t(&net, 100, 0.2, "inputs/hello.txt");
	for( int i = 0; i < 100000; i ++){
		t.train();
		std::string str = t.sample_string('\n', 100);
		std::cout << str << std::endl;
		if (!(i % 100 || true)){
			printf("Saving network...");
			std::ofstream fh;
			fh.open("outputs/saves/log.json");
			fh << net;
			fh.close();
			printf("Done!\n");
		}
			
	}
	
	
}
