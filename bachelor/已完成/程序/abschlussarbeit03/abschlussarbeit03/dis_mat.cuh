#ifndef DIS_MAT_CUH
#define DIS_MAT_CUH

#include "abschlussarbeit.cuh"

extern __global__ void matrixfix(unsigned char (*A)[NOV],
								 const unsigned char (*B)[NOV],
								 const unsigned char (*C)[NOV],
								 const unsigned char (*D)[NOV]);

extern __global__ void matrixMulCUDA(unsigned char (*C)[NOV],
									 unsigned char (*A)[NOV],
									 unsigned char (*B)[NOV]);

extern __global__ void computeDistanceMatrix(const unsigned int *prev, const unsigned int *last,
									  const unsigned short *other, const bool *isleaf,
									  float (*d)[NOV], float delta, 
									  //unsigned int (*stackp2d)[NOV], unsigned short (*stackk2d)[NOV], bool (*stackf2d)[NOV], bool (*visit2d)[NOV], 
									  bool (*left2d)[NOV], bool (*right2d)[NOV],
									  unsigned char (*a)[NOV]);

#endif /* DIS_MAT_CUH */

