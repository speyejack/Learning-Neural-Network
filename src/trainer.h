#ifndef TRAINER_HEADER_INCLUDED
#define TRAINER_HEADER_INCLUDED
#include "network.h"
#include <vector>

class Trainer{

private:
	int batch_size;
	int iteration;
	double learning_rate;
	std::vector<double> get_error(std::vector<double>& output, std::vector<double>& correct_output);
	Network* net;

protected:
	int get_batch_size(){return batch_size;}
public:
	Trainer(Network*, int, double);
	std::vector<double> train(std::vector<double>& input, std::vector<double>& correct_output);
};
#endif
