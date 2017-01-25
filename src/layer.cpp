#include "layer.h"
#include "matrix.h"
#include "vector.h"
#include <assert.h>
#include <ostream>

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
*/



Weights create_weights(int output_size, int input_size,
					   std::default_random_engine& gen,
					   double mean, double stddev){
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

Weights* create_empty_weights(int output_size, int input_size){
	Weights* w = new Weights();
	w->input = new Matrix(output_size, input_size);
	w->output = new Matrix(output_size, output_size);
 	w->memory = new Matrix(output_size, output_size);
	w->bias = new Matrix(output_size, 1);
	
	return w;
}

Weights* createWeightErrors(State* state, Matrix* error, Weights* prev_error){
	Vector bias(1);
	bias.set_value(0,0,1);
	
	Weights* weights = new Weights();
	// input weight is input->output size
	weights->input =
		new Matrix(error->dot(state->input->transpose()) +
				   *prev_error->input);
	// output weight is output->output size
	weights->output =
		new Matrix(error->dot(state->prev_state->output->transpose()) +
				   *prev_error->output);
	// memory weight is output->output size
	weights->memory =
		new Matrix(error->dot(state->memory->transpose()) +
				   *prev_error->memory);
	// bias weight is 1->output size
	weights->bias =
		new Matrix(error->dot(bias.transpose()) +
				   *prev_error->bias);
	return weights;
}

ErrorMatrix* createErrorMatrix(int size){
	ErrorMatrix* err = new ErrorMatrix();
	err->output = new Vector(size);
	err->memory = new Vector(size);
	return err;
}

void Layer::delete_weights(Weights w){
	delete w.input;
	delete w.output;
	delete w.memory;
	delete w.bias;
}

void deleteWeights(Weights* weights){
	if (weights == NULL)
		return;
	delete weights->input;
	delete weights->output;
	delete weights->memory;
	delete weights->bias;
	delete weights;
}

void deleteErrorMatrix(ErrorMatrix* errorMat){
	if (errorMat == NULL)
		return;
	delete errorMat->output;
	delete errorMat->memory;
	delete errorMat;
}

void deleteErrorState(ErrorState* eState){
	if (eState == NULL)
		return;
	deleteErrorMatrix(eState->error_input);
	deleteErrorMatrix(eState->error_forget);
	deleteErrorMatrix(eState->error_activate);
	deleteErrorMatrix(eState->error_output);
	delete eState->error_memory;
	delete eState->forget_gate;
	delete eState;
}

void deleteErrorOutput(ErrorOutput*);
void deleteErrorOutput(ErrorOutput* errorOut){
	if (errorOut == NULL)
		return;
	delete errorOut->inputError;
	deleteWeights(errorOut->input_werr);
	deleteWeights(errorOut->forget_werr);
	deleteWeights(errorOut->activate_werr);
	deleteWeights(errorOut->output_werr);
	ErrorOutput* next = errorOut->last;
	delete errorOut;
	deleteErrorOutput(next);
}

void deleteState(State*);
void deleteState(State* state){
	if (state == NULL)
		return;
	deleteErrorState(state->err_state);
	delete state->input;
	delete state->memory;
	delete state->output;
	delete state->input_gateP;
	delete state->forget_gateP;
	delete state->activation_gateP;
	delete state->output_gateP;
	State* next = state->prev_state;
	delete state;
	deleteState(next);
}

Layer::Layer(int input_size, int output_size, std::default_random_engine& gen){
	this->input_size = input_size;
	this->output_size = output_size;
	
	input_w = create_weights(output_size, input_size, gen, 0, 0.1);
	activate_w = create_weights(output_size, input_size, gen, 0, 0.1);
	forget_w = create_weights(output_size, input_size, gen, 0, 0.1);
	output_w = create_weights(output_size, input_size, gen, 0, 0.1);
	memory = new Vector(output_size);
	state = NULL;
	reset();
}

Layer::~Layer(){
	// I know, really bad memory leaks.
	// Just have to get the prop going first, then Ill go back.
	printf("Layer deconstructor broken");
}

Vector Layer::forward_prop(Vector& input){
	assert(input.get_height() == input_size);

	State* prev_state = state;
	
    state = new State();
	state->prev_state = prev_state;
	
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
	Matrix temp = forget_g * *memory + input_g * activation_g;
	memory = new Vector(temp);
	Vector activated_mem = memory->Mtanh();

	
	Matrix output_g_p = output_w.input->dot(input) +
		output_w.output->dot(*state->prev_state->output) +
		output_w.memory->dot(*memory) +
		output_w.bias->dot(bias);
	Matrix output_g = output_g_p.sigmoid();

	Vector output = (Vector) (output_g * activated_mem);

	
	state->input = new Vector(input);
	state->memory = new Vector(*memory);
	state->output = new Vector(output);
	state->input_gateP = new Matrix(input_g_p);
	state->forget_gateP = new Matrix(forget_g_p);
	state->activation_gateP = new Matrix(activation_g_p);
	state->output_gateP = new Matrix(output_g_p);
	state->err_state = NULL;
	
	return output;
}

ErrorOutput* Layer::back_prop(ErrorOutput* errorOut){
	State* top = state;
	// Initial blank err state
	ErrorState* errS = new ErrorState();
	errS->error_input = createErrorMatrix(this->output_size);
	errS->error_forget = createErrorMatrix(this->output_size);
	errS->error_activate = createErrorMatrix(this->output_size);
	errS->error_output = createErrorMatrix(this->output_size);
	errS->error_memory = new Vector(this->output_size);
	errS->forget_gate = new Matrix(state->forget_gateP->sigmoid());
	state->err_state = errS;
	
	ErrorOutput* out = apply_back_prop(errorOut);
	reset();
	return out;
}

ErrorOutput* Layer::apply_back_prop(ErrorOutput* errorOut){
	
	ErrorMatrix* error_input =  state->err_state->error_input;
	ErrorMatrix* error_forget =  state->err_state->error_forget;
	ErrorMatrix* error_activate =  state->err_state->error_activate;
	ErrorMatrix* error_output =  state->err_state->error_output;

	
	Matrix d_y = *errorOut->inputError +
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
		d_mem * *state->prev_state->memory *
		state->forget_gateP->sigDeriv();
	Matrix d_i =
		d_mem * state->activation_gateP->sigmoid() *
		state->input_gateP->sigDeriv();
	Matrix d_a =
		d_mem * state->input_gateP->sigmoid() *
		state->activation_gateP->sigDeriv();
	// Done with major back prop calculations
	
	
	ErrorState* err = new ErrorState();
	ErrorMatrix* err_in = new ErrorMatrix();
	
	err->error_input = err_in;
	err_in->memory = new Matrix(input_w.memory->transpose().dot(d_i));
	err_in->output = new Matrix(input_w.output->transpose().dot(d_i));
	
	ErrorMatrix* err_for = new ErrorMatrix();
	err->error_forget = err_for;
	err_for->memory = new Matrix(forget_w.memory->transpose().dot(d_f));
	err_for->output = new Matrix(forget_w.output->transpose().dot(d_f));
	
	ErrorMatrix* err_act = new ErrorMatrix();
	err->error_activate = err_act;
	err_act->memory = new Matrix(activate_w.memory->transpose().dot(d_a));
	err_act->output = new Matrix(activate_w.output->transpose().dot(d_a));

	ErrorMatrix* err_out = new ErrorMatrix();
	err->error_output = err_out;
	err_out->memory = new Matrix(input_w.memory->transpose().dot(d_o));
	err_out->output = new Matrix(input_w.output->transpose().dot(d_o));

	err->forget_gate = new Matrix(state->forget_gateP->sigmoid());
    err->error_memory = new Matrix(d_mem);
	state->prev_state->err_state = err;

	
	ErrorOutput* err_o = new ErrorOutput();
	
	if (state->prev_state->prev_state == NULL){
		ErrorOutput* lErr = new ErrorOutput();
		lErr->input_werr = create_empty_weights(output_size, input_size);
		lErr->forget_werr = create_empty_weights(output_size, input_size);
		lErr->activate_werr = create_empty_weights(output_size, input_size);
		lErr->output_werr = create_empty_weights(output_size, input_size);
		err_o->last = lErr;
	} else {
		State* cur_state = state;
		state = state->prev_state;
		// Recursive call
		err_o->last = apply_back_prop(errorOut->last);
		state = cur_state;
	}
	
    err_o->inputError =
		new Matrix(input_w.input->transpose().dot(d_i) +
				   forget_w.input->transpose().dot(d_f) +
				   activate_w.input->transpose().dot(d_a) +
				   output_w.input->transpose().dot(d_o));
	// The problem is the state->prev_state->output doesn't the correct size
	err_o->input_werr = createWeightErrors(state, &d_i,
										   err_o->last->input_werr);
	err_o->forget_werr = createWeightErrors(state, &d_f,
											err_o->last->forget_werr);
	err_o->activate_werr = createWeightErrors(state, &d_a,
											  err_o->last->activate_werr);
	err_o->output_werr = createWeightErrors(state, &d_o,
											err_o->last->output_werr);
	
	
	return err_o;
}

void Layer::clear_error(){
	// Changing how this error error prop works
}

void Layer::apply_error(double learning_rate){
	// Changing how this error error prop works
}

void Layer::reset(){
	deleteState(top);
	deleteErrorOutput(errorOut);
	state = NULL;
	top = NULL;
	errorOut = NULL;
	
	state = new State();
	state->output = new Vector(output_size);
	state->memory = new Vector(output_size);
	
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
std::ostream& operator<<(std::ostream& os, Layer& layer){
	layer.write_to_json(os);
	return os;
}
