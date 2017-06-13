#include "errorState.h"
#include "vector.h"

ErrorMatrix* createErrorMatrix(int size){
	ErrorMatrix* err = new ErrorMatrix();
	err->output = new Vector(size);
	err->memory = new Vector(size);

	return err;
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
