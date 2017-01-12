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

	state = NULL;
}

Layer::~Layer(){
	printf("Layer deconstructor broken");
}
void Layer::delete_state(){
	printf("Delete state broken");
}
/*

TODO: All this is broken

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
*/
void Layer::delete_weights(Weights w){
	delete w.input;
	delete w.output;
	delete w.memory;
	delete w.bias;
}

Vector Layer::forward_prop(Vector& input){
	assert(input.get_height() == input_size);

	State* prev_state = state;
	
    state = new State();
	
	Vector bias(1);
	bias.set_value(0,0,1);

	
	Matrix input_g_p = input_w.input->dot(input) +
		input_w.output->dot(*state->prev_state->output) +
		input_w.memory->dot(*memory) +
		input_w.bias->dot(bias);
	Matrix input_g = input_g_p.sigmoid();

	Matrix forget_g_p = forget_w.input->dot(input) +
		forget_w.output->dot(*state->prev_state->output) +
		forget_w.memory->dot(*memory) +
		forget_w.bias->dot(bias);
	Matrix forget_g = forget_g_p.sigmoid();

	Matrix activation_g_p = activate_w.input->dot(input) +
		activate_w.output->dot(*state->prev_state->output) +
		activate_w.bias->dot(bias);
	Matrix activation_g = activation_g_p.Mtanh();

	memory = new Vector(forget_g * *memory + input_g * activation_g);
	Vector activated_mem = memory->Mtanh();

	
	Matrix output_g_p = output_w.input->dot(input) +
		output_w.output->dot(*state->prev_state->output) +
		output_w.memory->dot(*memory) +
		output_w.bias->dot(bias);
	Matrix output_g = output_g_p.sigmoid();

	Vector output = (Vector) (output_g * activated_mem);

	
	state->prev_state = prev_state;
	state->memory = new Vector(*memory);
	state->output = new Vector(output);
	state->input_gateP = new Matrix(input_g_p);
	state->forget_gateP = new Matrix(forget_g_p);
	state->activation_gateP = new Matrix(activation_g_p);
	state->output_gateP = new Matrix(output_g_p);
	state->err_state = NULL;
	
	return output;
}

Vector Layer::back_prop(Vector& error){

	if (state->err_state == NULL){
		ErrorState* errS = new ErrorState();
		errS->error_input = createErrorMatrix(this->output_size);
		errS->error_forget = createErrorMatrix(this->output_size);
		errS->error_activate = createErrorMatrix(this->output_size);
	    errS->error_memory = new Vector(this->output_size);
		errS->forget_gate = new Matrix(state->forget_gateP->sigmoid());
		state->err_state = errS;
	}
	
	ErrorMatrix* error_input =  state->err_state->error_input;
	ErrorMatrix* error_forget =  state->err_state->error_forget;
	ErrorMatrix* error_activate =  state->err_state->error_activate;
	ErrorMatrix* error_output =  state->err_state->error_output;

	
	Matrix d_y = error +
		input_w.output->dot(*error_input->output) +
	    forget_w.output->dot(*error_forget->output) +
	    output_w.output->dot(*error_output->output) +
	    activate_w.output->dot(*error_activate->output);

	Matrix d_o = d_y * memory->Mtanh();
	Vector d_mem =
		d_y * memory->MtanhDeriv() *  state->output_gateP->sigmoid() +
		*state->err_state->forget_gate * *state->err_state->error_memory +
		input_w.output->dot(*error_input->memory) +
	    forget_w.output->dot(*error_forget->memory) +
	    output_w.output->dot(d_o);

	Matrix d_f =
		d_mem * *state->prev_state->memory * state->forget_gateP->sigDeriv();
	Matrix d_i =
		d_mem * state->activation_gateP->sigmoid() * state->input_gateP->sigDeriv();
	Matrix d_a =
		d_mem * state->input_gateP->sigmoid() * state->activation_gateP->sigDeriv();

	if (state->prev_state == NULL){
		return error;
	}

	ErrorState* err = new ErrorState();

	/*
	  fix this stuff
	err->error_input = new Matrix(d_i);
	err->error_forget = new Matrix(d_f);
	err->error_activate = new Matrix(d_a);
	err->error_memory = new Matrix(d_mem);
	*/
	err->forget_gate = new Matrix(state->forget_gateP->sigmoid());

	state->prev_state->err_state = err;
	
	return error; // TODO: put this to the sum of all input_i errors
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
