__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here

  // get global ids
  //  const int globalID_x = get_global_id(0);
  //  const int globalID_y = get_global_id(1);

  float sum;

  //do matrix multiplication

    for (int i = 0; i < numCRows; i++){ // iterate thru rows of C
      for(int j = 0; j < numCColumns; j++){ // iterate thru cols of C
        sum = 0.0;                          // reset sum for dot product
        for(int k = 0; k < numAColumns; k++){ // iterate thru columns of matrix A
          sum += A[numAColumns*i + k]*B[numCColumns*k +j];  // sum value 
        }
        C[i*numCColumns + j] = sum;                         // save value in matrix C
      }
    }



}