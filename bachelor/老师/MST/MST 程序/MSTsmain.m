clc;
close all;
clear all;
% ����1���� ����2����  ����3����
%����vleaf���������Ҷ�ӽڵ㣬leaf����Ҷ�ӽڵ�ֲ���leafnum���ں���
%rpconvert�еĻ����Ӽ�ɸѡ
disp('MSTs����');
disp('��ѡ����������һ������s��1��2��3��4��5��6��7��8');
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
%�������������󼰶���MST
D=distance(x,para1);
clear x;
if s<8
ST1=graphminspantree(D,'method','Kruskal');
ug2=D-ST1;
clear D;
ST2=graphminspantree(ug2,'method','Kruskal');
else
%����5000�ڵ������ʹ���Ա��Kruskal�㷨�������ڴ����
D=round(D*10000); %����5000�ڵ��������м��ʱ��ʹ�ø����
ST1=Krusk(D);
ug2=D-ST1;
clear D;
ST2=Krusk(ug2);
clear ug2;
ST1=ST1/10000;
ST2=ST2/10000;
end
%�������������Լ�overlapping
[commu_v,OL]=MSTs(ST1,ST2,para2,para3,leafnum);
%�ָ��ڵ�ԭʼ���
if leaf~=0
    deline=commu_v(end,:);
    for i=1:size(leaf,1)
        commu_v(commu_v>=leaf(i,2))=commu_v(commu_v>=leaf(i,2))+1;
        OL(OL>=leaf(i,2))=OL(OL>=leaf(i,2))+1;
    end
    commu_v(end,:)=deline;
end
%��������ʱ��
exctime=toc;
%********************��nmiֵ
if s>6
result=zeros(1,2);
result(2)=round(exctime);
nmi=GETNMI(community,commu_v,M);
result(1)=nmi;
end


