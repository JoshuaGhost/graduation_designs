#ifndef SEP_MST_CUH
#define SEP_MST_CUH

#include "abschlussarbeit.cuh"

extern __global__ void sep_mst(unsigned short *todeal,
							   unsigned int *prev, unsigned int *last, unsigned short *other, 
							   float *weight, bool (*inlgroup)[NOV]);

extern __global__ void reduct(bool (*inlgroup1)[NOV], bool (*inlgroup2)[NOV], bool *left, 
							  unsigned short epsilon, unsigned short theta, unsigned short pnum1);

extern __global__ void sep_mst2(unsigned short *todeal,
						unsigned int *prev, unsigned int *last, unsigned short *other, 
						float *weight, bool (*inlgroup)[NOV], unsigned short *group, 
						unsigned short rts);
#endif /* CUDA_SEPERATE_CUH */