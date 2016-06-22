#include "abschlussarbeit.cuh"

#define left left2d[tid]
#define oleft oleft2d[tid]

__global__ void computeDistanceMatrix(const unsigned int *prev, const unsigned int *last,
									  const unsigned short *other, const bool *isleaf,
									  float (*d)[NOV], float delta, 
									  bool (*left2d)[NOV], bool (*oleft2d)[NOV], unsigned char (*a)[NOV])
{
	unsigned short tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < NOV && !isleaf[tid]){
		for (unsigned short i = tid + 1; i < NOV; i++) if (!isleaf[i] && a[tid][i]<4) {
			for (unsigned int j = 0; j < NOV; j++)
				left[j] = oleft[j] = false;
			unsigned short num_left = 0;
			unsigned short num_right = 0;
			unsigned int ps = last[tid], pt = last[i];
			unsigned short t = 1;
			while (ps) {
				unsigned short otherps = other[ps];
				if (otherps != i){
					left[otherps] = oleft[otherps] = true;
					num_left++;
				}
				ps = prev[ps];
			}
			while (pt) {
				unsigned short otherpt = other[pt];
				if (otherpt == tid) {
					d[tid][i]+=1;
					pt = prev[pt];
					continue;
				}
				if (left[otherpt]) {
					d[tid][i]+=0.5;
					left[otherpt] = false;
					num_left--;
					pt = prev[pt];
					continue;
				}
				pt = prev[pt];
			}
			pt = last[i];
			while (pt) {
				bool isright = false;
				unsigned short otherpt = other[pt];
				if (!oleft[otherpt] && otherpt != tid){
					unsigned int p = last[otherpt];
					while (p) {
						unsigned short otherp = other[p];
						if (left[otherp]) {
							isright = true;
							left[otherp] = false;
							break;
						}
						p = prev[p];
					}
				}
				if (isright) {
					num_right++;
				}
				pt = prev[pt];
			}
			if (num_left != 0 && num_right != 0){
				unsigned int ans;
				if (num_left<num_right)
					ans = num_left;
				else
					ans = num_right;
				d[tid][i] += ans * 0.33333333333;				
			}
		}
		for (unsigned int i = tid+1; i < NOV; i++)
			d[i][tid] = d[tid][i];
		__syncthreads();
		for (unsigned int i = 0; i < NOV; i++)
			if (i != tid){
				if (d[tid][i]<0.00001 && d[tid][i]>-0.00001)
					d[tid][i] = 99999;
				else
					if (a[tid][i])
						d[tid][i] = (1/d[tid][i])*(1-delta);
					else
						d[tid][i] = 1/d[tid][i];
			}
	}
}

__global__ void matrixMulCUDA(unsigned char (*C)[NOV], 
							  unsigned char (*A)[NOV], 
							  unsigned char (*B)[NOV])
{
	unsigned int tid = blockDim.x * blockIdx.x +threadIdx.x;
	unsigned int cx = tid / NOV;
	unsigned int cy = tid % NOV;
	if (cx<NOV && cy<NOV){
		C[cx][cy] = 0;
#pragma unroll
		for (unsigned int i = 0; i < NOV; i++)
			C[cx][cy]|=A[cx][i]&B[i][cy];
	}
}

__global__ void matrixfix(unsigned char (*A), 
						  const unsigned char (*B),
						  const unsigned char (*C),
						  const unsigned char (*D))
{
	unsigned int tid = blockDim.x * blockIdx.x +threadIdx.x;
	unsigned int cx = tid / NOV;
	unsigned int cy = tid % NOV;
	if (cx >= NOV || cy >= NOV) return;
	if (cx == cy){
		A[tid] = 0;
		return;
	}
	if (B[tid]) {
		A[tid] = 1;
		return;
	}
	if (C[tid]) {
		A[tid] = 2;
		return;
	}
	if (D[tid]) {
		A[tid] = 3;
		return;
	}
	A[tid] = 99;
}

__global__ void Muld(unsigned char * A, unsigned char * B, int wA, int wB, unsigned char * C)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
 
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd = aBegin + wA - 1;
	int aStep = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * wB;
	
	unsigned char Csub = 0;
	for (int a = aBegin, b = bBegin;
	a <= aEnd;
	a += aStep, b += bStep) {
		if (b + wB * ty + tx < NOV * NOV){
			__shared__ unsigned char As[BLOCK_SIZE][BLOCK_SIZE];
			__shared__ unsigned char Bs[BLOCK_SIZE][BLOCK_SIZE];
			As[ty][tx] = A[a + wA * ty + tx];
			Bs[ty][tx] = B[b + wB * ty + tx];
			__syncthreads();
			for (int k = 0; k < BLOCK_SIZE; ++k)
				Csub += As[ty][k] * Bs[k][tx];
			__syncthreads();
		}
	}
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
} 