#ifndef NETWORK_HEADER_INCLUDED
#define NETWORK_HEADER_INCLUDED
#include <vector>
#include "layer.h"
#include <random>
#include <ostream>

class Network{
	
private:
	std::vector<Layer*> layers;
	std::seed_seq seed;
	ErrorOutput* errOut;
public:
	Network(std::vector<int>, std::seed_seq);
	~Network();
	std::vector<double> forward_prop(std::vector<double>& input);
	void back_prop(std::vector<double>& error);
	void apply_error(double learning_rate);
	void write_to_json(std::ostream& os);
};

std::ostream& operator<<(std::ostream& os, Network& net);
#endif
