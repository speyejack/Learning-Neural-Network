#ifndef LAYER_HEADER_INCLUDED
#define LAYER_HEADER_INCLUDED
#include "matrix.h"
#include "vector.h"
#include "weights.h"
#include <random>
#include <ostream>

// Struct to return from backprop
// Removing this out of redundency
/*
typedef struct ErrorOutput{ 
	ErrorOutput* last;
	Matrix* inputError;
	WeightBundle* error;
} ErrorOutput;
*/
typedef struct ErrorList{
    ErrorList* last;
	Matrix* error;
} ErrorList;

// Allows error to be held in errorState
typedef struct ErrorMatrix{
	Matrix* output;
	Matrix* memory;
} ErrorMatrix;

// Holds Error from next timestep (prev in backprop)
typedef struct ErrorState {
	WeightBundle* last_error;
	// From the next time step
	// Clean up this stuff for weight bundle instead
	ErrorMatrix* error_input;
	ErrorMatrix* error_forget;
	ErrorMatrix* error_activate;
	ErrorMatrix* error_output;
	Matrix* error_memory;
	Matrix* forget_gate;
} ErrorState;

// State of the cell
typedef struct State {
	State* prev_state;
	ErrorState* err_state;
	Vector* input;
	Vector* memory;
	Vector* output;
	Matrix* input_gateP;
	Matrix* forget_gateP;
	Matrix* activation_gateP;
	Matrix* output_gateP;
} State;


class Layer {
 private:
	int input_size;
	int output_size;

	WeightBundle* weights;
	WeightBundle* momentum;
	
	Vector* memory;
	State* state;
	
	void delete_state();
    int get_back_prop(ErrorList*, WeightBundle*, ErrorList* errIn);
	Weight applyWeightError(Weight, Weight*, Weight*, double, double);
	
 public:
	
	Layer(int input_size, int output_size, std::default_random_engine& gen);
	~Layer();
	int get_input_size(){return input_size;};
	int get_output_size(){return output_size;};
	
	Vector forward_prop(Vector& input);
	ErrorList* back_prop(ErrorList* errIn, double learning_rate);
	void reset();
	void write_to_json(std::ostream&);
};

std::ostream& operator<<(std::ostream&, Layer&);
#endif
