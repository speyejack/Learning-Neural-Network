#include "lstmLayer.h"
#include "matrix.h"
#include "vector.h"
#include "weights.h"
#include <assert.h>
#include <ostream>
// Remember to remove this after debugging...
#include <math.h>

void updateWeightErrors(Weight* matErr, State* state, Matrix* error){
	Vector bias(1);
	bias.set_value(0,0,1);
	
	
	// input weight is input->output size
	Matrix input = error->dot(state->input->transpose()) +
		*matErr->input;
	delete matErr->input;
	matErr->input = new Matrix(input);
	
	// output weight is output->output size
	Matrix output = error->dot(state->prev_state->output->transpose()) +
		*matErr->output;
	delete matErr->output;
	matErr->output = new Matrix(output);
	

	// memory weight is output->output size
	Matrix memory = error->dot(state->memory->transpose()) +
		*matErr->memory;
	delete matErr->memory;
	matErr->memory = new Matrix(memory);
	
	// bias weight is 1->output size
	Matrix bias_e = error->dot(bias.transpose()) +
		*matErr->bias;
	delete matErr->bias;
	matErr->bias = new Matrix(bias_e);
}

void deleteErrorList(ErrorList* list){
	if (list == NULL)
		return;
    ErrorList* next = list->last;
	delete list->error;
	delete list;
	deleteErrorList(next);
}

void adjustWeight(Weight* w, Weight* error, Weight* momentum, double learning_rate, double momentum_rate){
	Weight* new_mom = new Weight();
	Weight* new_w = new Weight();

	new_mom->input = new Matrix(*error->input * learning_rate + *momentum->input * momentum_rate);
	new_mom->output = new Matrix(*error->output * learning_rate + *momentum->output * momentum_rate);
	new_mom->memory= new Matrix(*error->memory * learning_rate + *momentum->memory * momentum_rate);
	new_mom->bias= new Matrix(*error->bias * learning_rate + *momentum->bias * momentum_rate);

	replaceWeight(momentum, new_mom);
	deleteWeight(new_mom);
	
	new_w->input = new Matrix(*w->input + *momentum->input);
	new_w->output = new Matrix(*w->output + *momentum->output);
	new_w->memory = new Matrix(*w->memory + *momentum->memory);
	new_w->bias = new Matrix(*w->bias + *momentum->bias);
	
	replaceWeight(w, new_w);
	deleteWeight(new_w);
}

LstmLayer::~LstmLayer(){
	deleteState(state);
	state = NULL;
	deleteWeightBundle(weights);
	deleteWeightBundle(momentum);
}

LstmLayer::LstmLayer(int input_size, int output_size, std::default_random_engine& gen): Layer(input_size, output_size){
	weights = createWeightBundle(getInputSize(), getOutputSize());
	momentum = createWeightBundle(getInputSize(), getOutputSize());
	fillBundle(weights, gen, 0, 0.1);
	
	state = NULL;
	reset();
}

Matrix getPrimitiveGate(Weight* weight, Vector* input, Vector* memory, Vector* output, Vector* bias){
	Matrix gate = weight->input->dot(*input) +
		weight->output->dot(*output) +
		weight->bias->dot(*bias);
	if (memory != NULL)
		gate = gate + weight->memory->dot(*memory);
	return gate;
}

Vector LstmLayer::forward_prop(Vector& input){
	assert(input.get_height() == getInputSize());

	Vector* memory = state->memory;
	
	State* prev_state = state;
	
    state = new State();
	state->prev_state = prev_state;
	
	Vector bias(1);
	bias.set_value(0,0,1);

	Matrix input_g_p =
		getPrimitiveGate(weights->input, &input,
						 memory, state->prev_state->output, &bias);
	Matrix input_g = input_g_p.sigmoid();

	Matrix forget_g_p =
		getPrimitiveGate(weights->forget, &input,
						 memory, state->prev_state->output, &bias);
	Matrix forget_g = forget_g_p.sigmoid();

	Matrix activation_g_p =
		getPrimitiveGate(weights->activate, &input,
						 NULL, state->prev_state->output, &bias);
	Matrix activation_g = activation_g_p.Mtanh();
	
	Matrix new_mem = forget_g * *memory + input_g * activation_g;
	
	memory = new Vector(new_mem);
	Vector activated_mem = memory->Mtanh();
	
	Matrix output_g_p =
		getPrimitiveGate(weights->output, &input,
						 memory, state->prev_state->output, &bias);
	Matrix output_g = output_g_p.sigmoid();

	Vector output = (Vector) (output_g * activated_mem);

	// Lots of assignments...
	state->input = new Vector(input);
	state->memory = new Vector(*memory);
	state->output = new Vector(output);
	state->input_gateP = new Matrix(input_g_p);
	state->forget_gateP = new Matrix(forget_g_p);
	state->activation_gateP = new Matrix(activation_g_p);
	state->output_gateP = new Matrix(output_g_p);
	state->err_state = NULL;
	delete memory;
	
	return output;
}

ErrorList* LstmLayer::back_prop(ErrorList* errIn, double learning_rate){
	State* top = state;
	// Initial blank err state
	ErrorState* errS = new ErrorState();
	errS->error_input = createErrorMatrix(getOutputSize());
	errS->error_forget = createErrorMatrix(getOutputSize());
	errS->error_activate = createErrorMatrix(getOutputSize());
	errS->error_output = createErrorMatrix(getOutputSize());
	errS->error_memory = new Vector(getOutputSize());
	errS->forget_gate = new Matrix(state->forget_gateP->sigmoid());
	state->err_state = errS;

	ErrorList* errOut = new ErrorList();
	WeightBundle* adjustments = createWeightBundle(getInputSize(), getOutputSize());
	int counter = get_back_prop(errOut, adjustments, errIn);

	ErrorList* topErr = errOut;
	errOut = errOut->last;
	delete topErr;

	double l_rate = learning_rate / counter;
	double momentum_rate = 0.5;
	
	adjustWeight(weights->input, adjustments->input, momentum->input, l_rate, momentum_rate);
	adjustWeight(weights->forget, adjustments->forget, momentum->forget, l_rate, momentum_rate);
	adjustWeight(weights->activate, adjustments->activate, momentum->activate, l_rate, momentum_rate);
	adjustWeight(weights->output, adjustments->output, momentum->output, l_rate, momentum_rate);

	deleteWeightBundle(adjustments);
	state = top;
	top = NULL;
	
	deleteErrorList(errIn);
	errIn = NULL;
	
	reset();
	return errOut;
}

int LstmLayer::get_back_prop(ErrorList* errOut, WeightBundle* weightErr, ErrorList* errIn){

	// If at end of chain, generate blank weight errors to return
	if (state->prev_state == NULL){
		assert(state->prev_state == NULL &&
			   errOut->last == NULL);
		return 0;
	}
	
	ErrorMatrix* error_input =  state->err_state->error_input;
	ErrorMatrix* error_forget =  state->err_state->error_forget;
	ErrorMatrix* error_activate =  state->err_state->error_activate;
	ErrorMatrix* error_output =  state->err_state->error_output;

	
	Matrix d_y = *errIn->error+
		weights->input->output->dot(*error_input->output) +
	    weights->forget->output->dot(*error_forget->output) +
	    weights->output->output->dot(*error_output->output) +
	    weights->activate->output->dot(*error_activate->output);

	Matrix d_o = d_y * state->memory->Mtanh();
	Vector d_mem =
		d_y * state->memory->MtanhDeriv() *  state->output_gateP->sigmoid() +
		*state->err_state->forget_gate * *state->err_state->error_memory +
		weights->input->output->dot(*error_input->memory) +
	    weights->forget->output->dot(*error_forget->memory) +
	    weights->output->output->dot(d_o);

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
	err_in->memory = new Matrix(weights->input->memory->transpose().dot(d_i));
	err_in->output = new Matrix(weights->input->output->transpose().dot(d_i));
	
	ErrorMatrix* err_for = new ErrorMatrix();
	err->error_forget = err_for;
	err_for->memory = new Matrix(weights->forget->memory->transpose().dot(d_f));
	err_for->output = new Matrix(weights->forget->output->transpose().dot(d_f));
	
	ErrorMatrix* err_act = new ErrorMatrix();
	err->error_activate = err_act;
	err_act->memory = new Matrix(weights->activate->memory->transpose().dot(d_a));
	err_act->output = new Matrix(weights->activate->output->transpose().dot(d_a));

	ErrorMatrix* err_out = new ErrorMatrix();
	err->error_output = err_out;
	err_out->memory = new Matrix(weights->input->memory->transpose().dot(d_o));
	err_out->output = new Matrix(weights->input->output->transpose().dot(d_o));

	err->forget_gate = new Matrix(state->forget_gateP->sigmoid());
    err->error_memory = new Matrix(d_mem);
	state->prev_state->err_state = err;


	ErrorList* err_o = new ErrorList();
	errOut->last = err_o;
    err_o->error =
		new Matrix(weights->input->input->transpose().dot(d_i) +
				   weights->forget->input->transpose().dot(d_f) +
				   weights->activate->input->transpose().dot(d_a) +
				   weights->output->input->transpose().dot(d_o));

	double prev_outin = weightErr->output->input->get_value(0,0);
	updateWeightErrors(weightErr->input, state, &d_i);
	updateWeightErrors(weightErr->forget, state, &d_f);
	updateWeightErrors(weightErr->activate, state, &d_a);
	updateWeightErrors(weightErr->output, state, &d_o);
	double outin = weightErr->output->input->get_value(0,0);
	// Remember to remove math.h
	if (fabs(outin - prev_outin) > 1000){
		printf("Diff of outin = %f\n", fabs(outin - prev_outin)); 
	}
	
	state = state->prev_state;
	
	return get_back_prop(err_o, weightErr, errIn->last) + 1;
}

void LstmLayer::reset(){
	deleteState(state);
	state = NULL;
	
	state = new State();
	state->output = new Vector(getOutputSize());
	state->memory = new Vector(getOutputSize());
}

void LstmLayer::write_to_json(std::ostream& os){
	os << "{" << std::endl;
	os << "\"input size\" : " << getInputSize() << ',' << std::endl;
	os << "\"output size\" : " << getOutputSize() << ',' << std::endl;
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
