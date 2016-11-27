#include <string>
#include "network.h"
#include "trainer.h"
class TextTrainer: public Trainer{
private:
	std::string file;
	int index;
	int file_size;
public:
	TextTrainer(Network*, int, double, std::string);
	void train();
	char sample(char);
};