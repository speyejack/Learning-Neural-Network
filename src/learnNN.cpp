#include <stdio.h>
#include "matrix.h"
#include <vector>
#include "gateTrainer.h"
#include "network.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>

int main(){
	std::vector<int> layers = {2, 1};
	unsigned long seed = time(0);
	std::cout << "Seed: " << seed << std::endl;
	seed = 1485472172;
	Network net(layers, {seed});
	
	GateTrainer t(&net, 1000, 0.01,
				  [](bool input1, bool input2) -> bool {
					  return input1 && input2;
				  });
	for( int i = 0; i < 100000; i ++){
		t.train();
		printf("Sampling...\n");
		for (int j = 0; j < 4; j++){
			// 10th run breaks
			double sample = t.sample(j/2, j%2);
			printf("%d && %d = %d : %.3f\n", j/2 , j%2, sample > 0.5, sample);
		}
		net.reset();
		/*
		  Taken out until it can be properly reimplemented
		if (!(i % 100 || true)){
			printf("Saving network...");
			std::ofstream fh;
			fh.open("outputs/saves/log.json");
			fh << net;
			fh.close();
			printf("Done!\n");
		}
		*/
	}
	
	
}
