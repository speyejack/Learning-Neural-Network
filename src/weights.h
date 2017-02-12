#ifndef WEIGHT_HEADER_INCLUDED
#define WEIGHT_HEADER_INCLUDED
#include "weights.h"
#include "matrix.h"
// Struct to hold each weight matrix
typedef struct Weight{
	Matrix* input;
	Matrix* output;
	Matrix* memory;
	Matrix* bias;
} Weight;

typedef struct WeightBundle{
	Weight* input;
	Weight* forget;
	Weight* activate;
	Weight* output;
} WeightBundle;


Weight* createEmptyWeight(int input_size, int output_size);

WeightBundle* createWeightBundle(int input_size, int output_size);

void deleteWeight(Weight* weight);

void deleteWeightBundle(WeightBundle* bundle);

void fillWeight(Weight* w, int output_size, int input_size,
					   std::default_random_engine& gen,
				  double mean, double stddev);

void fillBundle(WeightBundle* bundle,
				std::default_random_engine& gen,
				double mean, double stddev);
void replaceWeight(Weight* old_w, Weight* new_w);

void printWeight(Weight* weight);
void printBundle(WeightBundle* bundle);
#endif
