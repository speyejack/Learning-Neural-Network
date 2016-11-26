#include <stdio.h>
#include "matrix.h"
#include <vector>
#include "trainer.h"
#include "network.h"

int main(){
	std::vector<int> layers = {1, 1};
	Network n(layers, {1,1});
	Trainer t(&n, 2, 0.1);
	std::vector<double> input = {1};
	std::vector<double> true_output = {1};
	for(int i = 0; i < 10000; i++){
		t.train(input, true_output);
	}
	std::vector<double> output = n.forward_prop(input);
	Vector output_vec(output);
	printMatrix(output_vec);
}
