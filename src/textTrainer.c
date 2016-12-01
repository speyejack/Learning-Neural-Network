#include "textTrainer.h"
#include <vector>
#include <string>
#include <fstream>
#include <streambuf>
#include <ios>
#include <iostream>
#include "network.h"
#include "trainer.h"
#include "vector.h"

TextTrainer::TextTrainer(Network* net, int batch, double learning_rate, std::string filename): Trainer(net, batch, learning_rate){
	std::ifstream t(filename);
	file.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

	file_size = file.length();
	index = 0;
}

void TextTrainer::train(){
	std::vector<double> input(128, 0);
	std::vector<double> output(128, 0);
	
	if ( file_size < get_batch_size() + index + 2){
		index = 0;
	}
	
	for (int i = 0; i < get_batch_size(); i++){
		char char_loc = file[index + i];
		char next_char_loc = file[index + i];
		assert(char_loc >= 0);
		assert(next_char_loc >= 0);
		
		input[char_loc] = 1;
		output[next_char_loc] = 1;
		Trainer::train(input, output);
		input[char_loc] = 0;
		output[next_char_loc] = 0;
	}
	index++;
}

char TextTrainer::sample(char input, bool* small, double small_threshold){
	std::vector<double> input_v(128, 0);
	if (input != -1){
		input_v[input] = 1;
	}
	std::vector<double> output(128, 0);
	output = net->forward_prop(input_v);
	int top = 0;
	
	for (int i = 0; i < 128; i++){
		if (output[i] > output[top]){
			top = i;
		}
	}
	if (small != NULL){
		if (output[top] < small_threshold){
			*small = true;
		} else {
			*small = false;
		}
	}
	return top;
}

char TextTrainer::sample(char input){
	return sample(input, NULL, 0);
}


std::string TextTrainer::sample_string(char first, int size){
	std::string str("_", size);
	char c = first;
	char o = -1;
	bool small = false;
	for (int i = 0; i < size; i++){
		c = sample(c, &small, 0.3);
		
		if (small)
			o = 124;
		else if (c < 32)
			o = 95;
		else
			o = c;
		str[i] = o;
	}
	return str;
}
