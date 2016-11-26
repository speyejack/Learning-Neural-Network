#include "layer.h"
#include "matrix.h"
#include "vector.h"


Layer::Layer(int input_size, int output_size, std::default_random_engine gen){
	int true_input_size = input_size + output_size + 1;
	input_w = new Matrix(output_size, true_input_size);
	activate_w = new Matrix(output_size, true_input_size);
	forget_w = new Matrix(output_size, true_input_size);
	output_w = new Matrix(output_size, true_input_size);
	memory = new Vector(output_size);

	state = {0};
	
	state.prev_output = new Vector(output_size);

	input_w->fill_gaussian(gen, 0.0, 1.0);
	activate_w->fill_gaussian(gen, 0.0, 1.0);
	forget_w->fill_gaussian(gen, 0.0, 1.0);
	output_w->fill_gaussian(gen, 0.0, 1.0);
	
	error.err_input_w = new Matrix(output_size, true_input_size);
	error.err_activate_w = new Matrix(output_size, true_input_size);
	error.err_forget_w = new Matrix(output_size, true_input_size);
	error.err_output_w = new Matrix(output_size, true_input_size);

	this->input_size = input_size;
	this->output_size = output_size;
}

void Layer::clear_state(){
	delete state.prev_input;
	delete state.prev_output;
	delete state.prev_mem;
	delete state.input_gate;
	delete state.activate_gate;
	delete state.forget_gate;
	delete state.output_gate;
	delete state.activate_prim;
}

Vector Layer::forward_prop(Vector& input){
	Vector bias(1);
	bias.set_value(0,0,1);
	input = input.concatenate(*(state.prev_output)).concatenate(bias);
	
	clear_state();
	
	state.prev_input = new Vector(input);
	Matrix input_gate = input_w->dot(input).sigmoid();
	state.activate_prim = new Matrix(activate_w->dot(input));
	Matrix activate_gate = state.activate_prim->Mtanh();
	Matrix forget_gate = forget_w->dot(input).sigmoid();
	Matrix output_gate = output_w->dot(input).sigmoid();

	state.input_gate = new Matrix(input_gate);
	state.activate_gate = new Matrix(activate_gate);
	state.forget_gate = new Matrix(forget_gate);
	state.output_gate = new Matrix(output_gate);
	state.prev_mem = new Vector(*memory);

	*memory = (Vector) (input_gate * activate_gate + forget_gate * *memory);
	Vector tanhMem = *memory;
	tanhMem.Mtanh();
	Vector output = output_gate * tanhMem;
	output = output.subset(0, input_size);
	state.prev_output = new Vector(output);
	return output;
}

Vector Layer::back_prop(Vector& error){
	Vector tanhMem = *memory;
	tanhMem.Mtanh();
	Matrix del_output = error * tanhMem;
	Matrix del_mem = error * *(state.output_gate) * (((tanhMem * tanhMem) * -1.0) + 1.0);
	
	Matrix del_inputg = del_mem * *(state.activate_gate);
	Matrix del_forget = del_mem * *(state.prev_mem);
	Matrix del_activate = del_mem * *(state.input_gate);
	Matrix del_mem_prev = del_mem * *(state.forget_gate);

	Matrix tanhA = *(state.activate_prim);
	tanhA.Mtanh();
	
	Matrix del_activate_prim = del_activate * ((tanhA * -1) + 1);
	Matrix del_input_prim = del_inputg * *(state.input_gate) * (*(state.input_gate) * -1 + 1);
	Matrix del_forget_prim = del_forget* *(state.forget_gate) * (*(state.forget_gate) * -1 + 1);
	Matrix del_output_prim = del_output* *(state.output_gate) * (*(state.output_gate) * -1 + 1);

	Matrix del_input = input_w->transpose().dot(del_input_prim) +  activate_w->transpose().dot(del_activate_prim) + forget_w->transpose().dot(del_forget) + output_w->transpose().dot(del_output_prim);

	Matrix err_i_w = input_w->dot(state.prev_input->transpose());
	Matrix err_a_w = activate_w->dot(state.prev_input->transpose());
	Matrix err_f_w = forget_w->dot(state.prev_input->transpose());
	Matrix err_o_w = output_w->dot(state.prev_input->transpose());

	*(this->error.err_input_w) += err_i_w;
	*(this->error.err_activate_w) += err_a_w;
	*(this->error.err_forget_w) += err_f_w;
	*(this->error.err_output_w) += err_o_w;

	return ((Vector) del_input).subset(0, input_size);
}

void Layer::clear_error(){
	error.err_input_w->clear_matrix();
	error.err_activate_w->clear_matrix();
	error.err_forget_w->clear_matrix();
	error.err_output_w->clear_matrix();
}

void Layer::apply_error(double learning_rate){
	*input_w += (*(error.err_input_w) * learning_rate);
	*activate_w += (*(error.err_activate_w) * learning_rate);
    *forget_w += (*(error.err_forget_w) * learning_rate);
    *output_w += (*(error.err_output_w) * learning_rate);
	clear_error();
}
