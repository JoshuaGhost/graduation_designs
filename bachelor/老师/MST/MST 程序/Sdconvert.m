function Sd=Sdconvert(s,v,M,N,para2,leafnum)
%*******************************
% 输入：s MST的边集
%      v  对应边的距离大小
%      M  MST的节点数
%      N  MST的节点集
% 输出：Sd 划分子集序列
%*******************************计算最初的划分子集S
Sd=uint16(zeros(M+2,M-1));
for i=1:M-1
    c=liantong(s,M,i);  %移除1条边所得到的划分子集节点分类
    d=unique(c);
    c_1=N(c==d(1));
    c_2=N(c==d(2));
    if length(c_1)<=length(c_2)
        Sd(:,i)=[c_1;c_2;length(c_1);i];
    else
        Sd(:,i)=[c_2;c_1;length(c_2);i];
    end
end
%*******************************计算对应的叶子节点数
vnumber=zeros(M-1,1);
if leafnum~=0
    for i=1:M-1
        ta=setdiff(leafnum(:,1),Sd(1:Sd(M+1,i),i));
        tb=setdiff(leafnum(:,1),ta);
        leafv=0;
        for j=1:length(tb)
            leafv=leafv+leafnum(leafnum(:,1)==tb(j),2);
        end
        vnumber(i)=leafv;
    end
end
%*********************************计算划分子集序列Sε
for i=1:M-1
    if Sd(M+1,i)+vnumber(i)<=para2
        Sd(:,i)=uint16(zeros(M+2,1));
        v(i)=0;
    end
end
Sd=Sd(Sd~=0);
Sd=reshape(Sd,M+2,length(Sd)/(M+2));
v=v(v~=0);
%***************************计算最终的划分子集序列Sd
ave=mean(v);
for i=1:size(Sd,2)
    if v(i)<ave
        Sd(:,i)=uint16(zeros(M+2,1));
    end
end
Sd=Sd(Sd~=0);
Sd=reshape(Sd,M+2,length(Sd)/(M+2));
Sd=(sortrows(Sd',M+1))';