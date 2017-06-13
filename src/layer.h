#ifndef LAYER_HEADER_INCLUDED
#define LAYER_HEADER_INCLUDED
#include "matrix.h"
#include "vector.h"
#include <random>
#include <ostream>

// Struct to return from backprop
typedef struct ErrorList{
    ErrorList* last;
	Matrix* error;
} ErrorList;

class Layer {
 private:
	int input_size;
	int output_size;
	
 public:
	Layer(int input_size, int output_size) {this->input_size = input_size; this->output_size = output_size;};
	virtual ~Layer(){};
	int getInputSize(){return input_size;};
	int getOutputSize(){return output_size;};
	
	virtual Vector forward_prop(Vector& input) = 0;
	virtual ErrorList* back_prop(ErrorList* errIn, double learning_rate) = 0;
	virtual void reset() = 0;
	virtual void write_to_json(std::ostream&) = 0;
};

std::ostream& operator<<(std::ostream&, Layer&);
// Maybe move to it's own header file?
void deleteErrorList(ErrorList* list);
#endif
