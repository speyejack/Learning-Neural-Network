
void runAllMatrixTests();
void testCompareMatrixEqual();
void testMatrixGeneration();
void testMatrixGetHeight();
void testMatrixGetWidth();
void testMatrixGetValue();
void testMatrixSetValue();
void testMatrixTranspose();
void testMatrixClear();
void testMatrixDot();
Matrix generateMatrix(int* arrayMatrix, int height, int width);
void setMatrix(Matrix& matrix, int* arrayMatrix, int height, int width);
void setRow(Matrix& matrix, int* arrayRow, int row, int width);
Matrix generateAMatrix();
Matrix generateBMatrix();
bool compareMatrixEqual(Matrix& a, Matrix& b);
bool compareMatrixSizesEqual(Matrix& a, Matrix& b);
bool compareMatrixValuesEqual(Matrix& a, Matrix& b);
