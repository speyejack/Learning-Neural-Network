#ifndef GATE_TRAINER_HEADER_INCLUDED
#define GATE_TRAINER_HEADER_INCLUDED
#include "network.h"
#include "trainer.h"
#include <functional>

class GateTrainer: public Trainer{
private:
	std::function<bool(bool,bool)> gate;
public:
	GateTrainer(Network*, int, double, std::function<bool(bool, bool)>);
	void train();
	double sample(bool, bool);
};
#endif
