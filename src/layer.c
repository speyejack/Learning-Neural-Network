#include "layer.h"
#include "matrix.h"
#include "vector.h"
#include <assert.h>
#include <ostream>


Layer::Layer(int input_size, int output_size, std::default_random_engine& gen){
	this->input_size = input_size;
	this->output_size = output_size;
	
	input_w = create_weights(output_size, input_size, gen, 0, 0.1);
	activate_w = create_weights(output_size, input_size, gen, 0, 0.1);
	forget_w = create_weights(output_size, input_size, gen, 0, 0.1);
	output_w = create_weights(output_size, input_size, gen, 0, 0.1);
	memory = new Vector(output_size);

	state = {0, 0, 0, 0, 0, 0, 0, 0};
	
	state.prev_output = new Vector(output_size);
	state.prev_output->clear_matrix();
}

Layer::~Layer(){
	delete_state();
	delete_weights(forget_w);
	delete_weights(activate_w);
	delete_weights(input_w);
	delete_weights(output_w);
	delete memory;
	delete error.err_input_w;
	delete error.err_activate_w;
	delete error.err_forget_w;
	delete error.err_output_w;
}

void Layer::delete_state(){
	// May be changing structure of state
	delete state.prev_input;
	delete state.prev_output;
	delete state.prev_mem;
	delete state.input_gate;
	delete state.activate_gate;
	delete state.forget_gate;
	delete state.output_gate;
	delete state.activate_prim;
}

void Layer::delete_weights(Weights w){
	delete w.input;
	delete w.output;
	delete w.memory;
	delete w.bias;
}

Vector Layer::forward_prop(Vector& input){
	assert(input.get_height() == input_size);

	Vector bias(1);
	bias.set_value(0,0,1);
	
	Matrix input_g_p = input_w.input->dot(input) + input_w.output->dot(*state.prev_output) + input_w.memory->dot(*memory) + input_w.bias->dot(bias);
	Matrix input_g = input_g_p.sigmoid();

	Matrix forget_g_p = forget_w.input->dot(input) + forget_w.output->dot(*state.prev_output) + forget_w.memory->dot(*memory) + forget_w.bias->dot(bias);
	Matrix forget_g = forget_g_p.sigmoid();

	Matrix activation_g_p = activate_w.input->dot(input) + activate_w.output->dot(*state.prev_output) + activate_w.bias->dot(bias);
	Matrix activation_g = activation_g_p;
	activation_g.Mtanh();

	
	*memory = (Vector) (forget_g * *memory + input_g * activation_g);
	Vector activated_mem = *memory;
    activated_mem.Mtanh();

	
	Matrix output_g_p = output_w.input->dot(input) + output_w.output->dot(*state.prev_output) + output_w.memory->dot(*memory) + output_w.bias->dot(bias);
	Matrix output_g = output_g_p.sigmoid();

	Vector output = (Vector) (output_g * activated_mem);

	state.prev_output = new Vector(output);
	return output;
}

Vector Layer::back_prop(Vector& error){
	// Changing how back prop works
	return error;
}

void Layer::clear_error(){
	// Changing how this error error prop works
}

void Layer::apply_error(double learning_rate){
	// Changing how this error error prop works
}

void Layer::reset(){
	memory->clear_matrix();
}

void Layer::write_to_json(std::ostream& os){
	os << "{" << std::endl;
	os << "\"input size\" : " << input_size << ',' << std::endl;
	os << "\"output size\" : " << output_size << ',' << std::endl;
	os << "\"weights\" : {" << std::endl;
	
	// To make sure I fix this later
	fprintf(stderr, "Error: Re-implement write to json feature for layer\n");
	exit(1);
	
	/* Commented out to suppress compiler errors
	os << "\"input_w\" : " << *input_w << ',' << std::endl;
	os << "\"activate_w\" : " << *activate_w << ',' << std::endl;
	os << "\"forget_w\" : " << *forget_w << ',' << std::endl;
	os << "\"output_w\" : " << *output_w << ',' << std::endl;
	*/
	
	os << "}" << std::endl;
	os << "}";
}

Weights Layer::create_weights(int output_size, int input_size, std::default_random_engine& gen, double mean, double stddev){
	Weights w = {0, 0, 0, 0};
	w.input = new Matrix(output_size, input_size);
	w.output = new Matrix(output_size, output_size);
	w.memory = new Matrix(output_size, output_size);
	w.bias = new Matrix(output_size, 1);
	
	w.input->fill_gaussian(gen, mean, stddev);
	w.output->fill_gaussian(gen, mean, stddev);
	w.memory->fill_gaussian(gen, mean, stddev);
	w.bias->fill_gaussian(gen, mean, stddev);
	return w;
}

std::ostream& operator<<(std::ostream& os, Layer& layer){
	layer.write_to_json(os);
	return os;
}
