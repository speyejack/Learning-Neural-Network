#include "network.h"
#include "layer.h"
#include <assert.h>
#include <random>

Network::Network(std::vector<int> layer_sizes, std::seed_seq seed){
	// Not enough layers
	std::default_random_engine gen(seed);
	layers.resize(layer_sizes.size() - 1);
	assert(layer_sizes.size() > 1);
	Layer* l;
	for (unsigned int i = 1; i < layer_sizes.size(); i++){
		l = new Layer(layer_sizes[i - 1], layer_sizes[i], gen);
		layers[i-1] = l;
	}
}

Network::~Network(){
	for (unsigned int i = 0; i < layers.size(); i++){
		delete layers[i];
	}
}

std::vector<double> Network::forward_prop(std::vector<double>& input){
	Vector passing_vec(input);

	for(unsigned int i = 0; i < layers.size(); i++){
		passing_vec = layers[i]->forward_prop(passing_vec);
	}
	return passing_vec.to_std_vector();
}


void Network::back_prop(std::vector<double>& error){
	Vector error_vec(error);
	for(int i = layers.size() - 1; i >= 0; i--){
		error_vec = layers[i]->back_prop(error_vec);
	}
}

void Network::apply_error(double learning_rate){
	for(unsigned int i = 0; i < layers.size(); i++){
		layers[i]->apply_error(learning_rate);
		layers[i]->reset();
	}

}
