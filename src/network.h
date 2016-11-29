#ifndef NETWORK_HEADER_INCLUDED
#define NETWORK_HEADER_INCLUDED
#include <vector>
#include "layer.h"
#include <random>

class Network{
	
private:
	std::vector<Layer*> layers;
public:
	Network(std::vector<int>, std::seed_seq);
	~Network();
	std::vector<double> forward_prop(std::vector<double>& input);
	void back_prop(std::vector<double>& error);
	void apply_error(double learning_rate);
};
#endif
