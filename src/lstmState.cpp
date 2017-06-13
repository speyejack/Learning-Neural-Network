#include "lstmState.h"
#include "errorState.h"

void deleteState(State* state){
	if (state == NULL)
		return;
	State* next = state->prev_state;
	deleteErrorState(state->err_state);
	delete state->input;
	delete state->memory;
	delete state->output;
	delete state->input_gateP;
	delete state->forget_gateP;
	delete state->activation_gateP;
	delete state->output_gateP;
	delete state;
	deleteState(next);
}
