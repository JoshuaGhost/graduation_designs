#ifndef DIS_MAT_CUH
#define DIS_MAT_CUH

#include "abschlussarbeit.cuh"

extern __global__ void matrixfix(unsigned char (*A), 
						  const unsigned char (*B),
						  const unsigned char (*C),
						  const unsigned char (*D));

extern __global__ void matrixMulCUDA(unsigned char (*C)[NOV],
									 unsigned char (*A)[NOV],
									 unsigned char (*B)[NOV]);

extern __global__ void computeDistanceMatrix(const unsigned int *prev, const unsigned int *last,
									  const unsigned short *other, const bool *isleaf,
									  float (*d)[NOV], float delta, 
									  bool (*left2d)[NOV], bool (*right2d)[NOV],
									  unsigned char (*a)[NOV]);

extern __global__ void Muld(unsigned char * A, unsigned char * B, int wA, int wB, unsigned char * C);

#endif /* DIS_MAT_CUH */

