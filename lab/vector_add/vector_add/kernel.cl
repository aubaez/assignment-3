__kernel void vectorAdd(__global const float *a, __global const float *b,
                        __global float *result, const unsigned int size) {
  //@@ Insert code to implement vector addition here
  // get global id
  int globalID = get_global_id(0);
  //do addition of vectors
    result[globalID] = a[globalID] + b[globalID];
}