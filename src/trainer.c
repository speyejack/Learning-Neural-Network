#include <vector>

#include "trainer.h"

Trainer::Trainer(Network* network, int batch_size, double learning_rate){
    this->batch_size = batch_size;
	this->learning_rate = learning_rate;
	iteration = 0;
	
    // This may cause problems!
    net = network;
}

std::vector<double> Trainer::train(std::vector<double>& input, std::vector<double>& correct_output){
    std::vector<double> output = net->forward_prop(input);
    std::vector<double> error = get_error(output, correct_output);
    net->back_prop(error);
    if (iteration >= batch_size){
        net->apply_error(learning_rate);
		iteration = 0;
	} else {
		iteration++;
	}
	return output;
}

std::vector<double> Trainer::get_error(std::vector<double>& output, std::vector<double>& correct_output){
    std::vector<double> error(correct_output);
	
	for (int i = 0; i < error.size(); i++){
		error[i] -= output[i];
	}
	return error;
}
