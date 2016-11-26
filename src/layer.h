#ifndef LAYER_HEADER_INCLUDED
#define LAYER_HEADER_INCLUDED
#include "matrix.h"
#include "vector.h"
#include <random>

typedef struct State {
	Vector* prev_input;
	Vector* prev_output;
	Vector* prev_mem;
    Matrix* input_gate;
    Matrix* activate_gate;
	Matrix* forget_gate;
	Matrix* output_gate;
	Matrix* activate_prim;
}State;

typedef struct Error {
	Matrix* err_input_w;
	Matrix* err_activate_w;
	Matrix* err_forget_w;
	Matrix* err_output_w;
}Error;


class Layer{
private:
	int input_size;
	int output_size;
	Matrix* forget_w;
	Matrix* activate_w;
	Matrix* input_w;
	Matrix* output_w;
	Vector* memory;
	State state;
	Error error;
	void clear_error();
	void clear_state();
	
public:
	Layer(int input_size, int output_size, std::default_random_engine gen);
	int get_input_size(){return input_size;};
	int get_output_size(){return output_size;};
	
	Vector forward_prop(Vector& input);
	Vector back_prop(Vector& error);
	void apply_error(double learning_rate);
};

#endif
