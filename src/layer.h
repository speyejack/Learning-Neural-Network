#ifndef LAYER_HEADER_INCLUDED
#define LAYER_HEADER_INCLUDED
#include "matrix.h"
#include "vector.h"
#include <random>
#include <ostream>

typedef struct Weights{
	Matrix* input;
	Matrix* output;
	Matrix* memory;
	Matrix* bias;
}Weights;

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
	Weights* err_input_w;
    Weights* err_activate_w;
    Weights* err_forget_w;
    Weights* err_output_w;
}Error;


class Layer{
private:
	int input_size;
	int output_size;
	Weights forget_w;
    Weights activate_w;
    Weights input_w;
	Weights output_w;
	Vector* memory;
	State state;
	Error error;
	void clear_error();
	void delete_state();
	void delete_weights(Weights w);
	Weights create_weights(int, int, std::default_random_engine&, double, double);
	
public:
	
	Layer(int input_size, int output_size, std::default_random_engine& gen);
	~Layer();
	int get_input_size(){return input_size;};
	int get_output_size(){return output_size;};
	
	Vector forward_prop(Vector& input);
	Vector back_prop(Vector& error);
	void apply_error(double learning_rate);
	void reset();
	void write_to_json(std::ostream&);
};

std::ostream& operator<<(std::ostream&, Layer&);
#endif
