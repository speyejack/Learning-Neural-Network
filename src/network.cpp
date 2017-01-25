#include "network.h"
#include "layer.h"
#include <assert.h>
#include <random>
#include <ostream>
#include <iterator>

Network::Network(std::vector<int> layer_sizes, std::seed_seq seed){
	this->seed = seed;
	std::default_random_engine gen(seed);
	layers.resize(layer_sizes.size() - 1);
	// Not enough layers
	assert(layer_sizes.size() > 1);
	Layer* l;
	for (unsigned int i = 1; i < layer_sizes.size(); i++){
		l = new Layer(layer_sizes[i - 1], layer_sizes[i], gen);
		layers[i-1] = l;
	}
	errOut = NULL; 
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
	ErrorOutput* errO = new ErrorOutput();
	errO->inputError = new Vector(error);
	errO->last = errOut;
	errOut = errO;
}

void Network::apply_error(double learning_rate){
	for(int i = layers.size() - 1; i >= 0; i--){
		errOut = layers[i]->back_prop(errOut);
		//layers[i]->reset();
	}
	errOut = NULL;
}

void Network::write_to_json(std::ostream& os){
	os << "{" << std::endl;
	os << "\"Seed Size\" : " << seed.size() << "," << std::endl;
	std::ostream_iterator<unsigned> seed_out(os, " "); 
	os << "\"Seed\" : {"; seed.param(seed_out); os << "}," << std::endl;
	os << "\"Layers\" : " << layers.size() << "," << std::endl;
	for(unsigned int i = 0; i < layers.size(); i++){
		os << "\"Layer\" : " << *layers[i] << "," << std::endl ;
	}
	os << "}" << std::endl;
}

void Network::reset(){
	
	for(unsigned int i = 0; i < layers.size(); i++){
		layers[i]->reset();
	}
}

std::ostream& operator<<(std::ostream& os, Network& net){
	net.write_to_json(os);
	return os;
}
