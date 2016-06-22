#include "abschlussarbeit.cuh"

#define left left2d[tid]
#define right right2d[tid]

__device__ void findexroute(unsigned int tid,
							const unsigned short tk,
							const unsigned int *prev,
							const unsigned int *last,
							const unsigned short *other,
							bool *r,
							unsigned short *match,
							bool *res,
							unsigned int *stackp,
							unsigned short *stackk,
							bool *stackf,
							bool *visit)
{
	unsigned short k = tk;
	volatile unsigned int p = last[k];
	unsigned short sp = 0;
	
	for (unsigned int i = 0; i < NOV; i++){
		visit[i] = false;
		stackf[i] = stackk[i] = stackp[i] = 0;
	}

	while (p) {
		if (stackf[sp]) {
			match[other[p]] = k;
			if (sp) {
				sp--;
				p = stackp[sp];
				k = stackk[sp];
				stackf[sp] = true;
			}else{
				*res = true;
				return;
			}
		}else {
			if (r[other[p]] && !visit[other[p]]) {
				visit[other[p]] = true;
				if (match[other[p]] == 65535) {
					if (!sp) {
						match[other[p]] = k;
						*res = true;
						return;
					}else{
						sp--;
						match[other[p]] = k;
						p = stackp[sp];
						k = stackk[sp];
						stackf[sp] = true;
					}
				}else{
					stackp[sp] = p;
					stackk[sp] = k;
					sp++;
					k = match[other[p]];
					p = last[k];
				}
			}else{
				p = prev[p];
			}
		}
	}
	*res = false;
	return;
}

__global__ void computeDistanceMatrix(const unsigned int *prev, const unsigned int *last,
									  const unsigned short *other, const bool *isleaf,
									  float (*d)[NOV], float delta, 
									  //unsigned int (*stackp2d)[NOV], unsigned short (*stackk2d)[NOV], bool (*stackf2d)[NOV], bool (*visit2d)[NOV], 
									  bool (*left2d)[NOV], bool (*right2d)[NOV], unsigned char (*a)[NOV])
{
	unsigned short tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < NOV && !isleaf[tid]){
		for (unsigned short i = tid + 1; i < NOV; i++) if (!isleaf[i] && a[tid][i]<4) {
			for (unsigned int j = 0; j < NOV; j++)
				left[j] = right[j] = 0;
			unsigned short num_left = 0;
			unsigned short num_right = 0;
			unsigned int ps = last[tid], pt = last[i];
			unsigned short t = 1;
			while (ps) {
				unsigned short otherps = other[ps];
				if (otherps != i){
					left[otherps] = true;
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
				unsigned int p = last[otherpt];
				bool isright = false;
				while (p) {
					unsigned short otherp = other[p];
					if (left[otherp]) {
						isright = true;
						t++;
					}
					p = prev[p];
				}
				if (isright) {
					num_right++;
					right[otherpt] = true;
				}
				pt = prev[pt];
			}
			if (num_left != 0 && num_right != 0){
				if (num_left == 1 || num_right == 1) {
					d[tid][i] += 0.3333333;
				}else{
					unsigned int ans = 0;
					if (num_left<num_right)
						ans = num_left;
					else
						ans = num_right;
					d[tid][i] += ans * 0.3333333;
				}
			}
		}
		for (unsigned int i = tid+1; i < NOV; i++)
			d[i][tid] = d[tid][i];
		__syncthreads();
		for (unsigned int i = 0; i < NOV; i++)
			if (i != tid){
				if (d[tid][i]<0.00001 && d[tid][i]>-0.00001)
					d[tid][i] = 9999;
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

__global__ void matrixfix(unsigned char (*A)[NOV], 
						  const unsigned char (*B)[NOV],
						  const unsigned char (*C)[NOV],
						  const unsigned char (*D)[NOV])
{
	unsigned int tid = blockDim.x * blockIdx.x +threadIdx.x;
	unsigned int cx = tid / NOV;
	unsigned int cy = tid % NOV;
	if (cx >= NOV || cy >= NOV) return;
	if (cx == cy){
		A[cx][cy] = 0;
		return;
	}
	if (B[cx][cy]) {
		A[cx][cy] = 1;
		return;
	}
	if (C[cx][cy]) {
		A[cx][cy] = 2;
		return;
	}
	if (D[cx][cy]) {
		A[cx][cy] = 3;
		return;
	}
	A[cx][cy] = 99;
}

