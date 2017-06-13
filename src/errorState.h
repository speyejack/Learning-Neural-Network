#ifndef ERROR_STATE_HEADER_INCLUDED
#define ERROR_STATE_HEADER_INCLUDED
#include "matrix.h"
#include "weights.h"

// Allows error to be held in errorState
typedef struct ErrorMatrix{
	Matrix* output;
	Matrix* memory;
} ErrorMatrix;

// Holds Error from next timestep (prev in backprop)
typedef struct ErrorState {
	WeightBundle* last_error;
	// From the next time step
	// Clean up this stuff for weight bundle instead
	ErrorMatrix* error_input;
	ErrorMatrix* error_forget;
	ErrorMatrix* error_activate;
	ErrorMatrix* error_output;
	Matrix* error_memory;
	Matrix* forget_gate;
} ErrorState;

ErrorMatrix* createErrorMatrix(int size);
void deleteErrorMatrix(ErrorMatrix* errorMat);
void deleteErrorState(ErrorState* eState);
#endif
