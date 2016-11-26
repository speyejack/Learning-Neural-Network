#include <stdio.h>
#include "matrix.h"
#include <vector>
#include "trainer.h"
#include "network.h"

int main(){
	std::vector<int> layers = {1, 1};
	Network n(layers, {1,0});
	Trainer t(&n, 2, 0.01);
	std::vector<double> input = {1};
	std::vector<double> true_output = {1};
	for(int i = 0; i < 100; i++){
		t.train(input, true_output);
	}
}
