#include "gateTrainer.h"
#include "network.h"
#include "trainer.h"
#include <functional>
#include <vector>

GateTrainer::GateTrainer(Network* net, int batch, double learning_rate, std::function<bool(bool, bool)> gate): Trainer(net, batch, learning_rate), gate(gate){
	
}

void GateTrainer::train(){
	std::vector<double> input(2, 0);
	std::vector<double> output(1, 0);
	for (int i = 0; i < get_batch_size(); i++){
		input[0] = rand() % 2;
		input[1] = rand() % 2;
		output[0] = gate(input[0], input[1]);
		Trainer::train(input, output);
	}
	
}

double GateTrainer::sample(bool input1, bool input2){
	net->reset();
	std::vector<double> input(2, 0);
	std::vector<double> output(1, 0);
	input[0] = input1;
	input[1] = input2;
	output = net->forward_prop(input);
	
	return output[0];
}
