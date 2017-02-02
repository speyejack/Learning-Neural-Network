#include "weights.h"
#include "matrix.h"
// Struct to hold each weight matrix


Weight* createEmptyWeight(int input_size, int output_size){
	Weight* w = new Weight();
	w->input = new Matrix(output_size, input_size);
	w->output = new Matrix(output_size, output_size);
 	w->memory = new Matrix(output_size, output_size);
	w->bias = new Matrix(output_size, 1);
	
	return w;
}

WeightBundle* createWeightBundle(int input_size, int output_size){
	WeightBundle* bundle = new WeightBundle();
	bundle->input = createEmptyWeight(input_size, output_size);
	bundle->forget = createEmptyWeight(input_size, output_size);
	bundle->activate = createEmptyWeight(input_size, output_size);
	bundle->output = createEmptyWeight(input_size, output_size);
	return bundle;
}

void deleteWeight(Weight* weight){
	if (weight == NULL)
		return;
	delete weight->input;
	delete weight->output;
	delete weight->memory;
	delete weight->bias;
	delete weight;
}

void deleteWeightBundle(WeightBundle* bundle){
	deleteWeight(bundle->input);
	deleteWeight(bundle->forget);
	deleteWeight(bundle->activate);
	deleteWeight(bundle->output);
	delete bundle;
}

void fillWeight(Weight* w,
				   std::default_random_engine& gen,
				   double mean, double stddev){
	w->input->fill_gaussian(gen, mean, stddev);
	w->output->fill_gaussian(gen, mean, stddev);
	w->memory->fill_gaussian(gen, mean, stddev);
	w->bias->fill_gaussian(gen, mean, stddev);
}

void fillBundle(WeightBundle* bundle,
				std::default_random_engine& gen,
				double mean, double stddev){
	fillWeight(bundle->input, gen, mean, stddev);
	fillWeight(bundle->forget, gen, mean, stddev);
	fillWeight(bundle->activate, gen, mean, stddev);
	fillWeight(bundle->output, gen, mean, stddev);
}
