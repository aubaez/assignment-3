__kernel void matrixMultiply(
    __global const float *A, __global const float *B, __global float *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  

    const int tile_size = 16;

    __local float A_prime[tile_size][tile_size];
    __local float B_prime[tile_size][tile_size];
 
    int row_prime = get_local_id(0);
    int col_prime = get_local_id(1);

    int row = tile_size*get_group_id(0) + row_prime;
    int col = tile_size*get_group_id(1) + col_prime;

    const int num_tiles = (numAColumns+tile_size -1 )/tile_size;

    float sum = 0;

    for (int i; i < num_tiles; i++){
      int tile_row = tile_size*i + row_prime;
      int tile_col = tile_size*i + col_prime;

      // Populate A'
      if (row < numARows && col < numAColumns){
             A_prime[row][col] = A[row * numAColumns + tile_col];
      }
      else{
             A_prime[row][col] = 0;
      }
 

      // Populate B'
      if(row < numBRows && col < numBColumns){
        B_prime[row][col] = B[tile_row*numBColumns + col];
      }
      else{
        B_prime[row][col] = 0;
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for(int j = 0; j < tile_size; j++){
        sum += A_prime[row][j]*B_prime[j][col];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      
    }
    if(row < numARows && col < numBColumns){
      C[row * numBColumns + col] = sum; 
    }
    
}
