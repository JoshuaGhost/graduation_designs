#include "abschlussarbeit.cuh"
#include "dis_mat.cuh"
#include "min_reduction.cuh"
#include "sep_mst.cuh"

unsigned int n = NOV;
unsigned int m;
unsigned int *prev;
unsigned int *last;
unsigned short *other;
unsigned int rank;

unsigned int *prev1, *prev2;
unsigned int *last1, *last2;
unsigned short *other1, *other2;
float *weight1, *weight2;
float dslen1 = 0.0, dslen2 = 0.0;

bool (*inlgroup)[NOV];
bool *left;

unsigned short epsilon = 3, theta = 3;
float delta = 0.3;

unsigned short *fa;
unsigned int num_leaves = 0;
bool *isleaf;

float (*d)[NOV];
unsigned char (*a)[NOV];


class myException:public std::exception  
{  
public:  
    myException():exception("ERROR, Can't open file!\n")  
    {  
    }  
};

void readfile_edgelist()
{
	FILE *fp;
	try{
		fp = fopen("e:\\data_200x200x3820.txt","r");
		if (!fp)
			throw myException();
	}
	catch (myException& me){
		std::cout << me.what();
		system("pause");
		return;
	}
	fscanf(fp, "%d %d %f", &n, &m, &delta);
	for (unsigned int i = 1; i <= m; i++){
		unsigned short x, y, z;
		fscanf(fp,"%u %u %u", &x, &y, &z);
		x--;y--;
		prev[i] = last[x];
		last[x] = i;
		other[i] = y;
		prev[i + NOE] = last[y];
		last[y] = NOE + i;
		other[NOE + i] = x;
		a[x][y] = a[y][x] = 1;
	}
	fclose(fp);
}

void readfile_matrix()
{
	FILE *fp;
	try{
		fp = fopen("e:\\data_50x50x808.txt","r");
		if (!fp)
			throw myException();
	}
	catch (myException& me){
		std::cout << me.what();
		system("pause");
		return;
	}

	m = 0;
	int x;
	for (unsigned short i = 0; i < NOV; i++){
		a[i][i] = 0;
		for (unsigned short j = 0; j < NOV; j++){
			fscanf(fp, "%d", &x);
			if (x){
				m++;
				prev[m] = last[i];
				last[i] = m;
				other[m] = j;
				a[i][j] = (short)x;
			}
		}
	}
	fclose(fp);
}

void cudaMatrixMul()
{
	unsigned int size = NOV * NOV;
	unsigned int mem_size = sizeof(unsigned char) * size;
	unsigned char (*d_A)[NOV],(*d_B)[NOV],(*d_C)[NOV],(*d_D)[NOV];
    cudaMalloc((void **) &d_A, mem_size);
	cudaMalloc((void **) &d_B, mem_size);
	cudaMalloc((void **) &d_C, mem_size);
	cudaMalloc((void **) &d_D, mem_size);
	cudaMemcpy(d_A, a, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, d_A, mem_size, cudaMemcpyDeviceToDevice);

	unsigned int num_threads = (NOV*NOV)>1024 ? 1024 : NOV;
	unsigned int num_blocks = ((NOV*NOV)/num_threads) + (((NOV*NOV)%num_threads) ? 1 : 0);
	matrixMulCUDA<<<num_blocks, num_threads>>>(d_C, d_A, d_B);
	cudaDeviceSynchronize();
	matrixMulCUDA<<<num_blocks, num_threads>>>(d_D, d_C, d_B);
	cudaDeviceSynchronize();
	matrixfix<<<num_blocks, num_threads>>>(d_A, d_B, d_C, d_D);
	cudaDeviceSynchronize();
	cudaMemcpy(a, d_A, mem_size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_D);
}

unsigned short find(unsigned short p)
{
	if (fa[p] == p) 
		return p;
	fa[p] = find(fa[p]);
	return fa[p];
}

void search_leaves()
{
	unsigned short q[NOV];
	unsigned short head = 0, tail = 0;

	for (unsigned int i = 0; i < NOV; i++)
		fa[i] = i;

	for (int i = 0; i < NOV; i++)
		if (!prev[last[i]]) {
			q[head] = i;
			head++;
		}
	while (head > tail) {
		unsigned short num = 0;
		unsigned short father;
		unsigned int p = last[q[tail]];
		while (p){
			if (!isleaf[other[p]]){
				num++;
				father = other[p];
			}
			do {
			p = prev[p];
			}while (0);
		}
		if (num == 1){
			isleaf[q[tail]] = true;
			num_leaves++;
			fa[q[tail]] = find(father);
		}
		tail++;
	}
}

void cudaComputeDistanceMatrix()
{
	unsigned int *d_prev;
	unsigned int *d_last;
	unsigned short *d_other;
	bool *d_isleaf;
	float (*d_d)[NOV];
	bool (*d_left2d)[NOV];
	bool (*d_right2d)[NOV];
	unsigned char (*d_a)[NOV];

	cudaMalloc((void **)&d_prev, (2 * NOE + 2)*sizeof(unsigned int));
	cudaMalloc((void **)&d_last, NOV*sizeof(int));
	cudaMalloc((void **)&d_other, (2 * NOE + 2)*sizeof(unsigned short));
	cudaMalloc((void **)&d_isleaf, NOV*sizeof(bool));
	cudaMalloc((void **)&d_d, NOV*NOV*sizeof(float));
	cudaMalloc((void **)&d_right2d, NOV*NOV*sizeof(bool));
	cudaMalloc((void **)&d_left2d, NOV*NOV*sizeof(bool));
	cudaMalloc((void **)&d_a, NOV*NOV*sizeof(unsigned char));

	cudaMemcpy(d_d, d, NOV*NOV*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prev, prev, (2 * NOE + 2)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_last, last, NOV*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_other, other, (2 * NOE + 2)*sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaMemcpy(d_isleaf, isleaf, NOV*sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a, a, NOV*NOV*sizeof(unsigned char), cudaMemcpyHostToDevice);

	int num_threads = 1024, num_blocks = NOV / num_threads + (NOV % num_threads ? 1 : 0);
	computeDistanceMatrix<<<num_blocks, num_threads>>>(d_prev, d_last, d_other, d_isleaf, d_d, delta, 
													   d_left2d, d_right2d, d_a);
	cudaMemcpy(d, d_d, NOV*NOV*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%d\n",cudaGetLastError());
	cudaFree(d_prev);
	cudaFree(d_last);
	cudaFree(d_other);
	cudaFree(d_d);
}

void prim_mst(const unsigned int *v, const unsigned short *e, float *w, const unsigned int times)
{
	float *d_iweight = NULL, *d_oweight = NULL;
	unsigned int *d_oid = NULL;
	int num_threads = 1024;
	int num_blocks = NOV / num_threads+(NOV % num_threads == 0?0:1);
	cudaMalloc((void **)&d_iweight, n * sizeof(float));
	cudaMalloc((void **)&d_oid, sizeof(int));
	cudaMalloc((void **)&d_oweight, sizeof(float));

	unsigned int i, j;
	unsigned int ki;
	unsigned short k;
	unsigned int p = 1;
	unsigned int closest[NOV];
	float value[NOV] = {0};
	for (i = 1; i<n; i++)
		if (isleaf[i]) 
			value[i] = 0;
		else
			value[i] = MAX_WEIGHT;
	for (i = 0; i < NOV; i++)
		if (!isleaf[i])
			break;
	for (j = v[i]; j < v[i+1]; j++)
		value[e[j]] = w[j];
	value[i] = 0;
	for (j = 0; j < NOV; j++)
		closest[j] = i;

	for (i = 0; i < n-1-num_leaves; i++){
		
		cudaMemcpy(d_iweight, value, n * sizeof(float), cudaMemcpyHostToDevice);
		if (n == 5000){
			unsigned int resid[5];
			float resweight[5];
			for (unsigned j = 0; j < 5; j++)
				if (j < 4){
					cudaMemcpy(d_iweight, &value[j*1024], 1024*sizeof(float), cudaMemcpyHostToDevice);
					min_reduction1<<<1,num_threads>>>(d_iweight, d_oweight, d_oid, 1024);
					cudaMemcpy(&resweight[j], d_oweight, sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(&resid[j], d_oid, sizeof(int), cudaMemcpyDeviceToHost);
				} else {
					cudaMemcpy(d_iweight, &value[4096], 904*sizeof(float), cudaMemcpyHostToDevice);
					min_reduction1<<<1, num_threads>>>(d_iweight, d_oweight, d_oid, 904);
					cudaMemcpy(&resweight[j], d_oweight, sizeof(float), cudaMemcpyDeviceToDevice);
					cudaMemcpy(&resid[j], d_oid, sizeof(int), cudaMemcpyDeviceToHost);
				}
			unsigned int *d_iid = NULL;
			cudaMalloc((void **)&d_iid, 5*sizeof(int));
			cudaMemcpy(d_iid, resid, 5*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_iweight, resweight, 5*sizeof(float), cudaMemcpyHostToDevice);
			min_reduction2<<<1, 5>>>(d_iweight, d_oweight, d_iid, d_oid, 5);
		} else {
			min_reduction1<<<num_blocks, num_threads>>>(d_iweight, d_oweight, d_oid, n);
		}
		cudaMemcpy(&ki, d_oid, sizeof(int), cudaMemcpyDeviceToHost);
		k = (unsigned short)ki;
		if (times == 1){
			prev1[p] = last1[closest[k]];
			last1[closest[k]] = p;
			other1[p] = k;
			prev1[p + NOV - 1] = last1[k];
			last1[k] = p + NOV - 1;
			other1[p + NOV - 1] = closest[k];
			weight1[p] = value[k];
			weight1[p + NOV - 1] = weight1[p];
			dslen1 += value[k];
			for (j = v[closest[k]]; j< v[closest[k]+1]; j++){
				if (e[j] == k){
					w[j] = MAX_WEIGHT;
					break;
				}
			}
			for (j = v[k]; j < v[k+1]; j++){
				if (e[j] == closest[k]){
					w[j] = MAX_WEIGHT;
					break;
				}
			}
		}else{
			prev2[p] = last2[closest[k]];
			last2[closest[k]] = p;
			other2[p] = k;
			prev2[p + NOV - 1] = last2[k];
			last2[k] = p + NOV - 1;
			other2[p + NOV - 1] = closest[k];
			weight2[p] = value[k];
			weight2[p + NOV - 1] = weight2[p];
			dslen2 += value[k];
		}
		p++;
		value[k] = 0;
		for (j = v[k]; j < v[k+1]; j++)
			if (w[j] < value[e[j]]) {
				value[e[j]] = w[j];
				closest[e[j]] = k;
			}
	}
	cudaFree(d_iweight);
	cudaFree(d_oweight);
	cudaFree(d_oid);
	p = NOV - num_leaves;
	if (times == 1) {
		for (unsigned short i = 0; i < NOV; i++){
			if (fa[i] != i) {
				prev1[p] = last1[i];
				last1[i] = p;
				other1[p] = fa[i];
				prev1[p+NOV-1] = last1[fa[i]];
				last1[fa[i]] = p+NOV-1;
				other1[p+NOV-1] = i;
			}
		}
	}else{
		for (unsigned short i = 0; i < NOV; i++){
			if (fa[i] != i) {
				prev2[p] = last2[i];
				last2[i] = p;
				other2[p] = fa[i];
				prev2[p+NOV-1] = last2[fa[i]];
				last2[fa[i]] = p+NOV-1;
				other2[p+NOV-1] = i;
			}
		}
	}
}
/*
void cudaSeperate()
{
	unsigned int *d_prev = NULL, *d_last = NULL; 
	unsigned short *d_other = NULL;
	float *d_weight = NULL;
	bool (*d_inlgroup1)[NOV] = NULL, (*d_inlgroup2)[NOV] = NULL;

	bool inlgroupp[NOV][NOV] = {false};

	bool *d_left = NULL;
	unsigned short *d_todeal = NULL;
	unsigned short todeal[NOV] = {0};
	short num_deal1 = -1;
	for (unsigned short i = 1; i < NOV; i++)
		if (weight1[i]>dslen1){
			num_deal1++;
			todeal[num_deal1] = i;
		}
	cudaMalloc((void **)&d_todeal, sizeof(todeal));
	cudaMemcpy(d_todeal, todeal, sizeof(todeal), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_prev, (2*NOV-1)*sizeof(unsigned int));	
	cudaMemcpy(d_prev, prev1, (2*NOV-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_last, NOV*sizeof(unsigned int));
	cudaMemcpy(d_last, last1, NOV*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_other, (2*NOV-1)*sizeof(unsigned short));
	cudaMemcpy(d_other, other1, (2*NOV-1)*sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_inlgroup1, NOV*NOV*sizeof(bool));
	cudaMalloc((void **)&d_weight, (2*NOV-1)*sizeof(float));
	cudaMemcpy(d_weight, weight1, (2*NOV-1)*sizeof(float), cudaMemcpyHostToDevice);
	int num_threads = (num_deal1+1)>1024?1024:(num_deal1+1);
	int num_blocks = (num_deal1+1) / num_threads+((num_deal1+1) % num_threads == 0?0:1);
	sep_mst<<<num_blocks,num_threads>>>(d_todeal, d_prev, d_last, d_other, d_weight, d_inlgroup1);	
	cudaMemcpy(inlgroup, d_inlgroup1, NOV*NOV*sizeof(bool), cudaMemcpyDeviceToHost);

	short num_deal2 = -1;
	for (unsigned short i = 1; i < NOV; i++)
		if (weight2[i]>dslen1){
			num_deal2++;
			todeal[num_deal2] = i;
		}
	cudaMemcpy(d_todeal, todeal, sizeof(todeal), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prev, prev2, (2*NOV-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_last, last2, NOV*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_other, other2, (2*NOV-1)*sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight2, (2*NOV-1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_inlgroup2, NOV*NOV*sizeof(bool));
	num_threads = (num_deal2+1)>1024?1024:(num_deal2+1);
	num_blocks = (num_deal2+1) / num_threads+((num_deal2+1) % num_threads == 0?0:1);
	sep_mst<<<num_blocks,num_threads>>>(d_todeal, d_prev, d_last, d_other, d_weight, d_inlgroup2);

	cudaMemcpy(inlgroupp, d_inlgroup2, sizeof(inlgroupp), cudaMemcpyDeviceToHost);

	cudaFree(d_prev);
	cudaFree(d_last);
	cudaFree(d_other);
	cudaFree(d_weight);
	cudaMalloc((void **)&d_left, NOV*sizeof(bool));
	num_threads = (num_deal1+1)>1024?1024:(num_deal1+1);
	num_blocks = (num_deal1+1) / num_threads+((num_deal1+1) % num_threads == 0?0:1);
	reduct<<<num_blocks,num_threads>>>(d_inlgroup1, d_inlgroup2, d_left, epsilon, theta);
	cudaMemcpy(inlgroup, d_inlgroup1, NOV*NOV*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(left, d_left, NOV*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(d_left);
	cudaFree(d_inlgroup1);
	cudaFree(d_inlgroup2);

	unsigned short p = 1;
	FILE *fp;
	fp = fopen("e:\\data_out.txt", "w+");
	for (unsigned short i = 0; i <= num_deal1; i++)
		if (left[i]){
			fprintf(fp, "+++++++++++++++++++ %d +++++++++++++++++++\n", p);
			for (unsigned short j = 0; j < NOV; j++) {
				if (inlgroup[i][j]) {
					fprintf(fp, "%4d", j);
				}
			}
			fprintf(fp,"\n");
			for (unsigned short j = 0; j < NOV; j++){
				if (!inlgroup[i][j]) {
					fprintf(fp, "%4d", j);
				}
			}
			fprintf(fp,"\n");
			fprintf(fp,"------------------------------------------\n");
			p++;
		}
	fclose(fp);
}
*/
bool seperate(unsigned short *group, unsigned short rts)
{
	short t = -1;
	unsigned short *q;
	unsigned short head = 0, tail = 0;
	bool *v;
	float dslen1 = 0, dslen2 = 0;
	unsigned short pnum1 = 0, pnum2 = 0;
	bool etocut1[2 * NOV - 1] = {false}, etocut2[2 * NOV - 1] = {false};

	q = (unsigned short *)malloc(NOV*sizeof(short));
	v = (bool *)malloc((2*NOV-1)*sizeof(bool));
	memset(v,0,(2*NOV-1)*sizeof(bool));

	for (unsigned short i = 0; i < NOV; i++)
		if (group[i] == rts){
			t = i;
			break;
		}
	if (t == -1) return false;
	v[t] = true;
	q[tail] = t;
	while (head >= tail){
		unsigned int p;
		p = last1[q[tail]];
		while (p){
			if (!v[other1[p]] && group[other1[p]] == rts){
				v[other1[p]] = true;
				etocut1[p] = true;
				pnum1++;
				dslen1 += weight1[p];
				head++;
				q[head] = other1[p];
			}
			p = prev1[p];
		}
		tail++;
	}
	dslen1 /= pnum1;
	memset(v, 0, (2*NOV-1)*sizeof(bool));
	v[t] = true;
	head = tail = 0;
	while (head >= tail){
		unsigned short p;
		p = last2[q[tail]];
		while (p){
			if (!v[other2[p]] && group[other2[p]] == rts){
				v[other2[p]] = true;
				etocut2[p] = true;
				pnum2++;
				dslen2 += weight2[p];
				head++;
				q[head] = other2[p];
			}
			p = prev2[p];
		}
		tail++;
	}
	dslen2 /= pnum2;
	free(q);
	free(v);

	unsigned int *d_prev = NULL, *d_last = NULL; 
	unsigned short *d_other = NULL;
	float *d_weight = NULL;
	bool (*inlgroup1)[NOV];//, (*inlgroup2)[NOV];
	bool (*d_inlgroup1)[NOV] = NULL, (*d_inlgroup2)[NOV] = NULL;
	bool *d_left = NULL;
	unsigned short *d_todeal = NULL;
	unsigned short todeal1[NOV] = {0}, todeal2[NOV] = {0};
	short num_deal1 = -1;
	unsigned short *d_group = NULL;

	inlgroup1 = (bool (*) [NOV])malloc(NOV*NOV*sizeof(bool));
	//inlgroup2 = (bool (*) [NOV])malloc(NOV*NOV*sizeof(bool));

	for (unsigned short i = 1; i < NOV; i++)
		if (weight1[i]>dslen1 && etocut1[i]){
			num_deal1++;
			todeal1[num_deal1] = i;
		}
	if (num_deal1 < 0) return false;
	cudaMalloc((void **)&d_todeal, sizeof(todeal1));
	cudaMalloc((void **)&d_prev, (2*NOV-1)*sizeof(unsigned int));	
	cudaMalloc((void **)&d_last, NOV*sizeof(unsigned int));
	cudaMalloc((void **)&d_other, (2*NOV-1)*sizeof(unsigned short));
	cudaMalloc((void **)&d_inlgroup1, NOV*NOV*sizeof(bool));
	cudaMalloc((void **)&d_weight, (2*NOV-1)*sizeof(float));
	cudaMalloc((void **)&d_group, NOV * sizeof(short));
	cudaMemcpy(d_todeal, todeal1, sizeof(todeal1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prev, prev1, (2*NOV-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_last, last1, NOV*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_other, other1, (2*NOV-1)*sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight1, (2*NOV-1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_group, group, NOV * sizeof(short), cudaMemcpyHostToDevice);
	int num_threads = (num_deal1+1)>1024?1024:(num_deal1+1);
	int num_blocks = (num_deal1+1) / num_threads+((num_deal1+1) % num_threads == 0?0:1);
	sep_mst2<<<num_blocks,num_threads>>>(d_todeal, d_prev, d_last, d_other, d_weight, d_inlgroup1, d_group, rts);	
	cudaMemcpy(inlgroup1, d_inlgroup1, (num_deal1+1)*NOV*sizeof(bool), cudaMemcpyDeviceToHost);
	
	short num_deal2 = -1;
	for (unsigned short i = 1; i < NOV; i++)
		if (weight2[i]>dslen2 && etocut2[i]){
			num_deal2++;
			todeal2[num_deal2] = i;
		}
	if (num_deal2 < 0) return false;
	cudaMemcpy(d_todeal, todeal2, sizeof(todeal2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_prev, prev2, (2*NOV-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_last, last2, NOV*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_other, other2, (2*NOV-1)*sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight2, (2*NOV-1)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_inlgroup2, NOV*NOV*sizeof(bool));
	num_threads = (num_deal2+1)>1024?1024:(num_deal2+1);
	num_blocks = (num_deal2+1) / num_threads+((num_deal2+1) % num_threads == 0?0:1);
	sep_mst2<<<num_blocks,num_threads>>>(d_todeal, d_prev, d_last, d_other, d_weight, d_inlgroup2, d_group, rts);
	//cudaMemcpy(inlgroup2, d_inlgroup2, (num_deal2+1)*NOV*sizeof(bool), cudaMemcpyDeviceToHost);
	
	cudaFree(d_prev);
	cudaFree(d_last);
	cudaFree(d_other);
	cudaFree(d_weight);

	cudaMalloc((void **)&d_left, NOV*sizeof(bool));
	num_threads = (num_deal1+1)>1024?1024:(num_deal1+1);
	num_blocks = (num_deal1+1) / num_threads+((num_deal1+1) % num_threads == 0?0:1);
	reduct<<<num_blocks,num_threads>>>(d_inlgroup1, d_inlgroup2, d_left, epsilon, theta, pnum1);
	cudaMemcpy(left, d_left, NOV*sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(d_left);
	cudaFree(d_inlgroup1);
	cudaFree(d_inlgroup2);

	float max = 0;
	short fin = -1;
	for (unsigned short i = 0; i < num_deal1+1; i++)
		if (left[i] && weight1[todeal1[i]] > max){
			max = weight1[todeal1[i]];
			fin = i;
		}
	if (fin == -1) return false;
	rank++;

	for (unsigned short i = 0; i < NOV; i++)
		if (!inlgroup1[fin][i] && group[i] == rts)
			group[i] = rank;
	free(inlgroup1);
	return true;
	
}

void cudaSeperate2()
{
	unsigned short group[NOV] = {0};
	unsigned short q[2 * NOV] = {0};
	unsigned short head = 0, tail = 0;
	while (head >= tail){
		while (seperate(group, q[tail])){
			head++;
			q[head] = rank;
		}
		tail++;
	}

	FILE *fp;
	fp = fopen("e:\\data_out.txt", "w+");
	unsigned short p = 0, r = 1;
	bool v[NOV] = {false};
	while (p < NOV){
		unsigned short start = 0;
		fprintf(fp,"group %d:\n", r);
		while (v[start])
			start++;
		for (unsigned short i = start; i < NOV; i++)
			if (!v[i] && group[i] == group[start]){
				fprintf(fp,"%d\t\t", i);
				v[i] = true;
				p++;
			}
		fprintf(fp,"\n");
		r++;
	}
	fclose(fp);
}

int main()
{
	unsigned int *v;
	unsigned short *e;
	float *w;

	prev = (unsigned int *)malloc((2 * NOE + 2)*sizeof(unsigned int));
	last = (unsigned int *)malloc(NOV*sizeof(int));
	other = (unsigned short *)malloc((2 * NOE + 2)*sizeof(unsigned short));
	a = (unsigned char (*)[NOV])malloc(NOV*NOV*sizeof(unsigned char));
	memset(prev,0,(2 * NOE + 2)*sizeof(unsigned int));
	memset(last,0,NOV*sizeof(int));
	memset(other,0,(2 * NOE + 2)*sizeof(unsigned short));
	memset(a,0,NOV*NOV*sizeof(unsigned char));
	//readfile_edgelist();
	readfile_matrix();

	printf("%u\n",m);
	
	cudaMatrixMul();
	fa = (unsigned short *)malloc(NOV*sizeof(unsigned short));
	isleaf = (bool *)malloc(NOV*sizeof(bool));
	memset(fa,0,NOV*sizeof(unsigned short));
	memset(isleaf,0,NOV*sizeof(bool));

	search_leaves();

	d = (float (*)[NOV])malloc(NOV*NOV*sizeof(float));
	memset(d, 0, NOV*NOV*sizeof(float));

	cudaComputeDistanceMatrix();
	
	free(prev);
	free(last);
	free(other);
	free(a);

	v = (unsigned int *)malloc((NOV+1)*sizeof(unsigned int));
	e = (unsigned short *)malloc(NOV*NOV*sizeof(unsigned short));
	w = (float *)malloc(NOV*NOV*sizeof(float));
	memset(v,0,(NOV+1)*sizeof(unsigned int));
	memset(e,0,NOV*NOV*sizeof(unsigned short));
	memset(w,0,NOV*NOV*sizeof(float));

	unsigned int p, i;
	for (p = i = 0; i < n; i++){
		if (!isleaf[i]){
			v[i] = p;
			for (unsigned short j = 0; j < n; j++){
				if (!isleaf[j] && i!=j && d[i][j] < 998){
					e[p]=j;
					w[p]=d[i][j];
					p++;
				}
			}
		}
	}
	v[NOV - num_leaves] = p;
	free(d);

	prev1 = (unsigned int *)malloc((2*NOV-1)*sizeof(unsigned int));
	prev2 = (unsigned int *)malloc((2*NOV-1)*sizeof(unsigned int));
	last1 = (unsigned int *)malloc(NOV*sizeof(unsigned int));
	last2 = (unsigned int *)malloc(NOV*sizeof(unsigned int));
	other1 = (unsigned short *)malloc((2*NOV-1)*sizeof(unsigned short));
	other2 = (unsigned short *)malloc((2*NOV-1)*sizeof(unsigned short));
	weight1 = (float *)malloc((2*NOV-1)*sizeof(float));
	weight2 = (float *)malloc((2*NOV-1)*sizeof(float));
	memset(prev1,0,(2*NOV-1)*sizeof(unsigned int));
	memset(last1,0,NOV*sizeof(unsigned int));
	memset(other1,0,(2*NOV-1)*sizeof(unsigned short));
	memset(weight1,0,(2*NOV-1)*sizeof(float));
	memset(prev2,0,(2*NOV-1)*sizeof(unsigned int));
	memset(last2,0,NOV*sizeof(unsigned int));
	memset(other2,0,(2*NOV-1)*sizeof(unsigned short));
	memset(weight2,0,(2*NOV-1)*sizeof(float));

	prim_mst(v, e, w, 1);
	prim_mst(v, e, w, 2);
	free(fa);
	free(isleaf);

	dslen1 /= (NOV-1-num_leaves);
	dslen2 /= (NOV-1-num_leaves);

	inlgroup = (bool (*)[NOV])malloc(NOV*NOV*sizeof(bool));
	left = (bool *)malloc(NOV*sizeof(bool));
	memset(inlgroup,0,NOV*NOV*sizeof(bool));
	memset(left,0,NOV*sizeof(bool));

	//cudaSeperate();
	cudaSeperate2();

	free(left);
	free(inlgroup);
	free(prev1);
	free(prev2);
	free(last1);
	free(last2);
	free(other1);
	free(other2);
	free(weight1);
	free(weight2);
	
	return 0;
}

