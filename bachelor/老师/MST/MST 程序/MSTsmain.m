clc;
close all;
clear all;
% 参数1：δ 参数2：ε  参数3：θ
%函数vleaf计算网络的叶子节点，leaf给出叶子节点分布，leafnum便于函数
%rpconvert中的划分子集筛选
disp('MSTs……');
disp('请选择输入其中一个数字s：1、2、3、4、5、6、7、8');
s=input('s=');
switch s
    case 1
        load example1.mat
        para1=0;para2=3;para3=1;
    case 2
        load example2.mat
        para1=0;para2=3;para3=1;
    case 3
        load dolphin.mat
        para1=0;para2=3;para3=2;
    case 4
        load polbooks.mat
        para1=0.2;para2=3;para3=2;
    case 5
        load example3.mat
        para1=0;para2=3;para3=2;
    case 6
        load example4.mat
        para1=0;para2=3;para3=2;
    case 7
        load lfr1000.mat
        load lfrcommunity1000.mat
        para1=0;para2=3;para3=2;
    case 8
        load lfr5000.mat
        load lfrcommunity5000.mat
        para1=0;para2=3;para3=2;
end
tic;
M=length(x);
[leaf,leafnum]=vleaf(x);
if leaf~=0
    x(leaf(:,2),:)=[];
    x(:,leaf(:,2))=[];
end
%计算网络距离矩阵及二层MST
D=distance(x,para1);
clear x;
if s<8
ST1=graphminspantree(D,'method','Kruskal');
ug2=D-ST1;
clear D;
ST2=graphminspantree(ug2,'method','Kruskal');
else
%对于5000节点的网络使用自编的Kruskal算法，避免内存溢出
D=round(D*10000); %当对5000节点的网络进行检测时，使用该语句
ST1=Krusk(D);
ug2=D-ST1;
clear D;
ST2=Krusk(ug2);
clear ug2;
ST1=ST1/10000;
ST2=ST2/10000;
end
%求得社区检测结果以及overlapping
[commu_v,OL]=MSTs(ST1,ST2,para2,para3,leafnum);
%恢复节点原始序号
if leaf~=0
    deline=commu_v(end,:);
    for i=1:size(leaf,1)
        commu_v(commu_v>=leaf(i,2))=commu_v(commu_v>=leaf(i,2))+1;
        OL(OL>=leaf(i,2))=OL(OL>=leaf(i,2))+1;
    end
    commu_v(end,:)=deline;
end
%程序运行时间
exctime=toc;
%********************求nmi值
if s>6
result=zeros(1,2);
result(2)=round(exctime);
nmi=GETNMI(community,commu_v,M);
result(1)=nmi;
end


