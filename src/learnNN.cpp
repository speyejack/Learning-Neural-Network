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
	std::vector<int> layers = {2, 1};
	unsigned long seed = time(0);
	std::cout << "Seed: " << seed << std::endl;
	Network net(layers, {seed});
	
	GateTrainer t(&net, 1, 0.2,
				  [](bool input1, bool input2) -> bool {
					  return input1 && input2;
				  });
	for( int i = 0; i < 100000; i ++){
		t.train();
		printf("Sampling...\n");
		for (int j = 0; j < 4; j++){
			printf("%d && %d = %d\n", j/2 , j%2, t.sample(j/2, j%2));
		}
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
