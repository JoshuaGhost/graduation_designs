#include "sep_mst.cuh"

__global__ void sep_mst(unsigned short *todeal,
						unsigned int *prev, unsigned int *last, unsigned short *other, 
						float *weight, bool (*inlgroup)[NOV])
{
	bool l[NOV] = {false};
	unsigned short tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned short td = todeal[tid];
	unsigned short q[NOV] = {0};
	unsigned short head = 0;
	unsigned short tail = 0;
	l[0] = true;
	while (head >= tail){
		unsigned int p = last[q[tail]];
		while (p){
			if (p != td && p != td + NOV - 1){
				unsigned short otherp = other[p];
				if (!l[otherp]){
					head++;
					q[head] = otherp;
					l[otherp] = true;
				}
			}
			p = prev[p];
		}
		tail++;
	}
	for (unsigned i = 0; i < NOV; i++)
		inlgroup[tid][i] = l[i];
}

__global__ void reduct(bool (*inlgroup1)[NOV], bool (*inlgroup2)[NOV], bool *left, 
					   unsigned short epsilon, unsigned short theta, unsigned short pnum1)
{
	unsigned int tid = blockDim.x *blockIdx.x + threadIdx.x;
	left[tid] = false;
	bool l1[NOV] = {false};
	unsigned short sum = 0;
	for (unsigned short i = 0; i < NOV; i++){
		if (inlgroup1[tid][i]){
			l1[i] = true;
			sum++;
		}
	}
	if (sum >= epsilon && pnum1-sum>=epsilon){
		for (unsigned short i = 0; i < NOV; i++){
			unsigned short t = 0;
			while (t < NOV && !inlgroup2[i][t]) t++;
			if (t == NOV) {
				left[tid] = false;
				continue;
			} else {
				bool legal = true;
				unsigned char dismatch = 0;
				for (unsigned short j = t; j < NOV; j++){
					if (l1[j] ^ inlgroup2[i][j]){
						dismatch++;
					}
				}
				if (dismatch < theta || pnum1 - dismatch < theta){
					left[tid] = true;
					break;
				}
			}
		}
	} else {
		left[tid] = false;
	}
}

__global__ void sep_mst2(unsigned short *todeal,
						unsigned int *prev, unsigned int *last, unsigned short *other, 
						float *weight, bool (*inlgroup)[NOV], unsigned short *group, unsigned short rts)
{
	bool l[NOV] = {false};
	unsigned short tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned short td = todeal[tid];
	unsigned short q[NOV] = {0};
	unsigned short head = 0;
	unsigned short tail = 0;
	for (unsigned short i = 0; i < NOV; i++)
		if (group[i] == rts){
			l[i] = true;
			q[tail] = i;
			break;
		}
	while (head >= tail){
		unsigned int p = last[q[tail]];
		while (p){
			if (p != td && p != td + NOV - 1){
				unsigned short otherp = other[p];
				if (!l[otherp] && group[otherp] == rts){
					head++;
					q[head] = otherp;
					l[otherp] = true;
				}
			}
			p = prev[p];
		}
		tail++;
	}
	for (unsigned i = 0; i < NOV; i++)
		inlgroup[tid][i] = l[i];
}