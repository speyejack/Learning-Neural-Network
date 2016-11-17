#include <matrix.h>
class BlockMatrix(){

 private:
	std::vector<Matrix*> matrices;
	
	int height;
	int width;

 public:
	BlockMatrix(int width, int height);
	~BlockMatrix();
	BlockMatrix dot(BlockMatrix& b);
}
