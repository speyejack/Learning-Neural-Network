#ifndef LSTM_STATE_HEADER_INCLUDED
#define LSTM_STATE_HEADER_INCLUDED
#include "matrix.h"
#include "vector.h"
#include "errorState.h"

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

void deleteState(State*);
#endif
