#ifndef ABSCHLUSSARBEIT_CUH
#define ABSCHLUSSARBEIT_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <exception>
#include <iostream>
#include <fstream>

#define INPUTFILE "e:\\data_5000x5000x100940.txt"
#define OUTPUTFILE "e:\\data_out_5000x5000x100940.txt"
#define NOE 100940
#define NOV 5000
#define MAX 1025
#define MAX_WEIGHT 9999999
#define BLOCK_SIZE 16

/*
//n:��ĸ�����m:�ߵĸ���
unsigned short n = NOV;
unsigned int m;
unsigned int prev[2 * NOE + 2] = {0};
unsigned int last[NOV] = {0};
unsigned short other[2 * NOE + 2] = {0};
//����С���������С�����������0��ʼ�������ߴ�1��ʼ��
unsigned int prev2[2*NOV-1] = {0},prev1[2*NOV-1] = {0};
unsigned int last2[NOV] = {0},last1[NOV] = {0};
unsigned short other2[2*NOV-1] = {0},other1[2*NOV-1] = {0};
float weight1[2*NOV-1] = {0.0}, weight2[2*NOV-1] = {0.0};
//��/��С�������߳�ƽ��ֵ��
float dslen1 = 0.0, dslen2 = 0.0;
//�ָ���С��������ʱ�򶥵��Ƿ���0������ͬһ�飺
bool inlgroup[NOV][NOV];
//������µķ��鷽����
bool left[NOV];
unsigned short epsilon, theta;
float delta;
//Ҷ�ӽڵ�Ķ����ת������
unsigned int num_leaves = 0;
unsigned short lson[NOV] = {0}, rson[NOV] = {0};
bool isleaf[NOV] = {0};
unsigned char a[NOV][NOV] = {-1};
//distance matrix
float d[NOV][NOV] = {0};
*/

#endif /* ABSCHLUSSARBEIT_CUH */