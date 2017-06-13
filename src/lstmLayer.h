#ifndef LSTM_LAYER_HEADER_INCLUDED
#define LSTM_LAYER_HEADER_INCLUDED
#include "layer.h"
#include "matrix.h"
#include "vector.h"
#include "weights.h"
#include "lstmState.h"
#include <random>
#include <ostream>

class LstmLayer: public Layer {
 private:
	WeightBundle* weights;
	WeightBundle* momentum;
	
	State* state;
	
	void delete_state();
    int get_back_prop(ErrorList*, WeightBundle*, ErrorList* errIn);
	Weight applyWeightError(Weight, Weight*, Weight*, double, double);
	
 public:
	
	LstmLayer(int input_size, int output_size, std::default_random_engine& gen);
	~LstmLayer();

	Vector forward_prop(Vector& input);
	ErrorList* back_prop(ErrorList* errIn, double learning_rate);
	void reset();
	void write_to_json(std::ostream&);
};

std::ostream& operator<<(std::ostream&, Layer&);
// Maybe move to it's own header file?
void deleteErrorList(ErrorList* list);
#endif
