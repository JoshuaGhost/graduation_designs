#ifndef MIN_REDUCTION_CUH
#define MIN_REDUCTION_CUH

#include "abschlussarbeit.cuh"

extern __global__ void min_reduction1(float *d_iweight, float *d_oweight,
									 unsigned int *d_oid, unsigned int n);

extern __global__ void min_reduction2(float *d_iweight, float *d_oweight,
							  unsigned int *d_oid, unsigned int *d_iid,
							  unsigned int n);

#endif /* PRIM_MST_CUH */

