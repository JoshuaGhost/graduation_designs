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
//n:点的个数，m:边的个数
unsigned short n = NOV;
unsigned int m;
unsigned int prev[2 * NOE + 2] = {0};
unsigned int last[NOV] = {0};
unsigned short other[2 * NOE + 2] = {0};
//最新小生成树与次小生成树，点从0开始计数，边从1开始：
unsigned int prev2[2*NOV-1] = {0},prev1[2*NOV-1] = {0};
unsigned int last2[NOV] = {0},last1[NOV] = {0};
unsigned short other2[2*NOV-1] = {0},other1[2*NOV-1] = {0};
float weight1[2*NOV-1] = {0.0}, weight2[2*NOV-1] = {0.0};
//最/次小生成树边长平均值：
float dslen1 = 0.0, dslen2 = 0.0;
//分割最小生成树的时候顶点是否与0顶点在同一组：
bool inlgroup[NOV][NOV];
//最后留下的分组方案：
bool left[NOV];
unsigned short epsilon, theta;
float delta;
//叶子节点的多叉树转二叉树
unsigned int num_leaves = 0;
unsigned short lson[NOV] = {0}, rson[NOV] = {0};
bool isleaf[NOV] = {0};
unsigned char a[NOV][NOV] = {-1};
//distance matrix
float d[NOV][NOV] = {0};
*/

#endif /* ABSCHLUSSARBEIT_CUH */