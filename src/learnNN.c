#include <stdio.h>
#include "matrix.h"
#include <vector>
#include "trainer.h"
#include "network.h"
#include <cstdlib>
#include <ctime>

char index_to_char(int index){
	switch(index){
	case 0:
		return 'H';
	case 1:
		return 'e';
	case 2:
		return 'l';
	case 3:
		return 'o';
	case 4:
		return ' ';
	}
}
int char_to_index(char c){
	switch(c){
	case 'H':
		return 0;
	case 'e':
		return 1;
	case 'l':
		return 2;
	case 'o':
		return 3;
	case ' ':
		return 4;
	}
}

char get_vec_index_at_str_index(std::string str, int index){
	char c;
	if (index >= str.length())
		c = ' ';
	else
		c = str[index];
	return char_to_index(c);

}

int main(){
	std::vector<int> layers = {5, 5, 5};
	Network n(layers, {1,1});
	Trainer t(&n, 10, 0.1);
	
	std::string str = "Hello";
	std::string out_str = "     ";
	std::vector<double> input;
	std::vector<double> true_output;
	input.resize(5);
	true_output.resize(5);
    
	for(int i = 0; i < 100000; i++){
		for (int j = 0; j < str.length(); j++){
			
			int index = i % str.length() + j;

		    int input_i = get_vec_index_at_str_index(str, index);
		    int output_i = get_vec_index_at_str_index(str, index + 1);
			input[input_i] = 1;
			true_output[output_i] = 1;
			t.train(input, true_output);
			input[input_i] = 0;
			true_output[output_i] = 0;
		}

		if (i % 50000 == 0){
			for (int j = 0; j < str.length(); j++){
			
				int index = i % str.length() + j;

				int input_i = get_vec_index_at_str_index(str, index);
				input[input_i] = 1;
				std::vector<double> output = n.forward_prop(input);
				
				Vector input_vec(input);
				Vector output_vec(output);
				printf("Vect: \n");
				printMatrix(input_vec);
				printMatrix(output_vec);

				input[input_i] = 0;
			}
			printf("\n");
		}
	}
}
