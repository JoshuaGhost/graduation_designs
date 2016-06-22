#include "min_reduction.cuh"

__device__ void warpReduce1(volatile float *sdata, volatile unsigned int *sid, float *ssweight, unsigned int *ssid,
							unsigned int tid, unsigned int blockSize, unsigned int n) 
{
	if (blockSize >= 64) if (sdata[tid] > sdata[tid + 32] && tid+32<n) {
		*ssweight = sdata[tid] = sdata[tid + 32];
		*ssid = sid[tid] = sid[tid+32];
	}
	if (blockSize >= 32) if (sdata[tid] > sdata[tid + 16] && tid+16<n) {
		*ssweight = sdata[tid] = sdata[tid + 16];
		*ssid = sid[tid]= sid[tid+16];
	}

	if (blockSize >= 16) if (sdata[tid] > sdata[tid + 8] && tid+8<n)  {
		*ssweight = sdata[tid] = sdata[tid + 8];
		*ssid = sid[tid] = sid[tid+8];
	}
	if (blockSize >= 8)  if (sdata[tid] > sdata[tid + 4] && tid+4<n)  {
		*ssweight = sdata[tid] = sdata[tid + 4];
		*ssid = sid[tid] = sid[tid+4];
	}
	if (blockSize >= 4)  if (sdata[tid] > sdata[tid + 2] && tid+2<n)  {
		*ssweight = sdata[tid] = sdata[tid + 2];
		*ssid = sid[tid]= sid[tid+2];
	}
	if (blockSize >= 2)  if (sdata[tid] > sdata[tid + 1] && tid+1<n)  {
		*ssweight = sdata[tid] = sdata[tid + 1];
		*ssid = sid[tid] = sid[tid+1];
	}
}

__device__ void warpReduce2(volatile float *sdata, volatile unsigned int *sid, float *ssweight, unsigned int *ssid,
							unsigned int tid, unsigned int blockSize, unsigned int n) 
{
	if (blockSize >= 64) if (sdata[tid] > sdata[tid + 32] && tid+32<n) {
		*ssweight = sdata[tid] = sdata[tid + 32];
		*ssid = sid[tid] = sid[tid+32];
	}
	if (blockSize >= 32) if (sdata[tid] > sdata[tid + 16] && tid+16<n) {
		*ssweight = sdata[tid] = sdata[tid + 16];
		*ssid = sid[tid]= sid[tid+16];
	}

	if (blockSize >= 16) if (sdata[tid] > sdata[tid + 8] && tid+8<n)  {
		*ssweight = sdata[tid] = sdata[tid + 8];
		*ssid = sid[tid] = sid[tid+8];
	}
	if (blockSize >= 8)  if (sdata[tid] > sdata[tid + 4] && tid+4<n)  {
		*ssweight = sdata[tid] = sdata[tid + 4];
		*ssid = sid[tid] = sid[tid+4];
	}
	if (blockSize >= 4)  if (sdata[tid] > sdata[tid + 2] && tid+2<n)  {
		*ssweight = sdata[tid] = sdata[tid + 2];
		*ssid = sid[tid]= sid[tid+2];
	}
	if (blockSize >= 2)  if (sdata[tid] > sdata[tid + 1] && tid+1<n)  {
		*ssweight = sdata[tid] = sdata[tid + 1];
		*ssid = sid[tid] = sid[tid+1];
	}
}

__global__ void min_reduction1(float *d_iweight, float *d_oweight,
							  unsigned int *d_oid, unsigned int n) 
{
	__shared__ float sdata[1024];
	__shared__ unsigned int sid[1024];
	unsigned int tid = threadIdx.x;
	unsigned int blockSize = blockDim.x;
	unsigned int i = blockIdx.x * blockSize + tid;
	unsigned int gridSize = gridDim.x * blockDim.x;
	float ssweight;
	unsigned int ssid;
	sdata[tid] = 0;
	sid[tid] = 0;
	if ((i < gridSize) && (i >= n))
		ssweight = sdata[tid] = MAX_WEIGHT;
	if (i < n){
		ssweight = sdata[tid] = d_iweight[i];
		ssid = sid[i] = i;
		if (sdata[tid] == 0)
			ssweight = sdata[tid] = MAX_WEIGHT;
	}
	__syncthreads();
	if (blockSize == 1024) {
		if ((tid < 512) && (sdata[tid] > sdata[tid + 512]) && tid+512<n){
			ssweight = sdata[tid] = sdata[tid+512];
			ssid = sid[tid] = sid[tid+512];
		}
		__syncthreads();
	}
	if (blockSize >= 512) {
		if ((tid < 256) && (sdata[tid] > sdata[tid + 256]) && tid+256<n) {
			ssweight = sdata[tid] = sdata[tid + 256];
			ssid = sid[tid] = sid[tid+ 256];
		}
		__syncthreads(); 
	}
	if (blockSize >= 256) { 
		if ((tid < 128) && (sdata[tid] > sdata[tid + 128]) && tid+128<n) {
			ssweight = sdata[tid] = sdata[tid + 128]; 
			ssid = sid[tid] = sid[tid+128];
		}
		__syncthreads(); 
	}
	if (blockSize >= 128) { 
		if ((tid < 64) &&  (sdata[tid] > sdata[tid + 64]) && tid+64<n){
			ssweight = sdata[tid] = sdata[tid + 64]; 
			ssid = sid[tid] = sid[tid + 64];
		}
		__syncthreads(); 
	}
	if (tid < 32) warpReduce1(sdata, sid, &ssweight, &ssid, tid, blockSize, n);
	if (tid == 0) {
		*d_oid = sid[0];
		*d_oweight = sdata[0];
	}
}

__global__ void min_reduction2(float *d_iweight, float *d_oweight,
							  unsigned int *d_oid, unsigned int *d_iid,
							  unsigned int n) 
{
	__shared__ float sdata[5];
	__shared__ unsigned int sid[5];
	unsigned int tid = threadIdx.x;
	unsigned int blockSize = blockDim.x;
	unsigned int i = blockIdx.x * blockSize + tid;
	unsigned int gridSize = gridDim.x * blockDim.x;

	sdata[tid] = d_iweight[i];
	sid[i] = d_iid[i];
	if (sdata[tid] == 0)
		sdata[tid] = MAX_WEIGHT;

	if (blockSize >= 4)  if (sdata[tid] > sdata[tid + 2] && tid+2<n)  {
		sdata[tid] = sdata[tid + 2];
		sid[tid]= sid[tid+2];
	}
	if (blockSize >= 2)  if (sdata[tid] > sdata[tid + 1] && tid+1<n)  {
		sdata[tid] = sdata[tid + 1];
		sid[tid] = sid[tid+1];
	}
	*d_oid= sid[0];
	*d_oweight= sdata[0];
}
