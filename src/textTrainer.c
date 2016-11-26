#include "textTrainer.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <ios>
#include "network.h"
#include "trainer.h"

TextTrainer::TextTrainer(Network* net, int batch, double learning_rate, std::string filename): Trainer(net, batch, learning_rate){
	std::ifstream t(filename);
	std::stringstream buffer;
	buffer << t.rdbuf();

	file = buffer.str();
	file_size = file.length();
	index = 0;
}

void TextTrainer::train(){
	std::vector<double> input;
	input.resize(128);
	std::vector<double> output;
	output.resize(128);
	for (int i = 0; i < get_batch_size(); i++){
		input[file[index + i]] = 1;
		output[file[index + i + 1]] = 1;
		Trainer::train(input, output);
		input[file[index + i]] = 0;
		output[file[index + i + 1]] = 0;
	}
	if (get_batch_size() + index >= file_size){
		index = 0;
	}
}

char TextTrainer::sample(char input){
	std::vector<double> input_v;
	input_v.resize(128);
	if (input != -1){
		input_v[input] = 1;
	}
	std::vector<double> output;
	output = net->forward_prop(input_v);
	for (int i = 0; i < 128; i++){
		if (output[i] > 0.95)
			return i;

	}
	return 0;
	
}
