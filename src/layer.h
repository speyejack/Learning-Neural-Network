

class Layer{


private:
	int input_size;
	int output_size;
	Matrix* forget_w;
	Matrix* activate_w;
	Matrix* input_w;
	Matrix* output_w;
	Matrix* memory;
	
public:
	Layer(int input_size, int output_size);
	int get_input_size();
	int get_output_size();
	
	Matrix& forward_prop(Matrix& input);
	Matrix& back_prop(Matrix& error);
}

	struct State{
		
	}State;
