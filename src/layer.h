#ifndef LAYER_HEADER_INCLUDED
#define LAYER_HEADER_INCLUDED
#include "matrix.h"
#include "vector.h"
#include <random>
#include <ostream>
// THOUGHT: Structs in structs?
// Struct to hold each weight matrix
typedef struct Weights{
	Matrix* input;
	Matrix* output;
	Matrix* memory;
	Matrix* bias;
} Weights;

// Struct to return from backprop
typedef struct ErrorOutput{ 
	ErrorOutput* last;
	Matrix* inputError;
    Weights* input_werr;
	Weights* forget_werr;
    Weights* activate_werr;
	Weights* output_werr;
} ErrorOutput;

// Allows error to be held in errorState
typedef struct ErrorMatrix{
	Matrix* output;
	Matrix* memory;
} ErrorMatrix;


// Holds Error from next timestep (prev in backprop)
typedef struct ErrorState {
	// From the next time step
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
	Weights forget_w;
    Weights activate_w;
    Weights input_w;
	Weights output_w;
	Vector* memory;
	State* state;
	void clear_error();
	void delete_state();
	void delete_weights(Weights w);
	ErrorOutput* get_back_prop(ErrorOutput* error);
	Weights applyWeightError(Weights, Weights*, Weights*, double);
	
 public:
	
	Layer(int input_size, int output_size, std::default_random_engine& gen);
	~Layer();
	int get_input_size(){return input_size;};
	int get_output_size(){return output_size;};

	
	Vector forward_prop(Vector& input);
ErrorOutput* back_prop(ErrorOutput* error);
	void apply_error(double learning_rate);
	void reset();
	void write_to_json(std::ostream&);
};

std::ostream& operator<<(std::ostream&, Layer&);
#endif
