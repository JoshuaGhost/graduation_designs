function [commu_v,OL]=MSTs(ST1,ST2,para2,para3,leafnum)
%*****************************************
%输入：ST1 ST2 二层最小生成树；
%     para2 参数ε，根据公式7筛选划分子集Sd，并移除相似性划分
%     para3 参数θ，划分中允许存在的最大相异节点数，用公式15处理
%     leafnum 对应根节点包含的叶子节点个数 
%输出：commu_v 社区检测结果，每一列为一个社区
%     OL 社区之间的overlapping
%*****************************************
M=uint16(length(ST1));             %移除叶子节点后的网络节点数
MAX=uint16(1000);                  %网络所有的有效划分子集最大值，可以节省内存空间，防止溢出
sub1_v=uint16(zeros(M,MAX));       %来自第一层MST有效划分后的节点子集
sub1_v(:,1)=1:M;                   %初始为网络的所有节点
sub2_v=uint16(zeros(M,MAX));       %来自第二层MST有效划分后的节点子集
sub2_v(:,1)=1:M;                   %初始为网络的所有节点
subol_v=uint16(zeros(M,MAX));        %来自不同MST有效划分的节点子集之间的overlapping
commu_n=uint16(zeros(500,1));      %最终确定为社区的节点子集，
subx=1;            %当前进行步骤3-7的节点子集
suby=1;            %sub_v中当前划分出的节点子集数
commux=1;          %当前划分出的社区数
OLpr=[];           %移除相似性划分得到的overlapping
%*******************************************
while any(sub1_v(:,subx))      %判断所有的节点子集是否都通过步骤3-7处理
    if M==53&&subx>1
        para3=1;
    end
    N1=sub1_v(1:sub1_v(end,subx),subx);   %由于存在overlapping，sub1_v和sub2_v对应的节点子集包含的节点数不相同，
    N2=sub2_v(1:sub2_v(end,subx),subx);   %需要分别确定各自包含的节点N1和N2
    M1=length(ST1(N1,N1));
    M2=length(ST2(N2,N2));
    if subol_v(end,subx)~=0
        OL=subol_v(1:subol_v(end,subx),subx);
    else
        OL=[];
    end
    %*********************************
    [a1,b1,v1]=find(ST1(N1,N1));           %选取当前结点子集对应的最小生成树
    [a2,b2,v2]=find(ST2(N2,N2));
    s1=[a1 b1];
    s2=[a2 b2];
    clear a1 a2 b1 b2;
    %*********************************
    Sd1=Sdconvert(s1,v1,M1,N1,para2,leafnum); %计算2个划分子集Sd1和Sd2
    Sd2=Sdconvert(s2,v2,M2,N2,para2,leafnum);
    %*********************************计算所有满足条件15的划分子集
    n1=size(Sd1,2);n2=size(Sd2,2);
    cc=zeros(n2,n1);             
    for i=1:n1
        for j=1:n2 
            dif_a=Diff(Sd1(:,i),Sd2(:,j),M1,M2,OL); %Diff求得相异节点
            if length(dif_a)<=para3
                cc(j,i)=length(dif_a)+1;
            end
        end
    end
    if isempty(cc(cc~=0))
        cpa=[];
    else
    cc=sparse(cc);
    [cp01,cp02,vsita]=find(cc);
    if size(cp01,2)>1
        cp01=cp01';
        cp02=cp02';
        vsita=vsita';
    end
    cc1=[cp01 cp02 vsita];
    cc1=sortrows(cc1,3);
    %*********************************初步筛选相似性划分
    t=1;
    ccp=zeros(length(vsita),2);  %筛选后保留下的有效划分
    ccp(1,:)=cc1(1,1:2);
    for i=2:length(vsita)
        k=0;
        for j=1:t
           if cc1(i,1)==ccp(j,1)
               k=1;
               Sdc1=Sd1(:,cc1(i,2));
               Sdc2=Sd1(:,ccp(j,2));
               dif_pr=Diff(Sdc1,Sdc2,M1,M1,[]);
               OLpr=cat(1,OLpr,dif_pr);
           elseif cc1(i,2)==ccp(j,2)
               k=1;
               Sdc1=Sd2(:,cc1(i,1));
               Sdc2=Sd2(:,ccp(j,1));
               dif_pr=Diff(Sdc1,Sdc2,M2,M2,[]);
               OLpr=cat(1,OLpr,dif_pr);
           end
        end
        if k==0
            ccp(t+1,:)=cc1(i,1:2);
            t=t+1;
        end
    end
    %****************************对ccp中的有效划分再次筛选相似性划分
    cpa=ccp(1:t,2)';
    cpb=ccp(1:t,1)';
    end
    cp1=[];cp2=[];%最终的有效划分子集
    if ~isempty(cpa)
        cp01=Sd1(:,cpa);
        cp02=Sd2(:,cpb);
        cp1(:,1)=cp01(:,1);
        t=1;t1=2;
        pick=zeros(1,size(cp01,2));
        pick(1)=1;
        if size(cp01,2)==1
            cp2=cp02;
        else
            for i=2:size(cp01,2)
                for j=1:t
                    t2=0;
                    dif_a=Diff(cp01(:,i),cp1(:,j),M1,M1,OL);
                    if length(dif_a)<=para2
                        t2=1;
                        dif_pr=Diff(cp01(:,i),cp02(:,i),M1,M2,OL);
                        if ~isempty(dif_pr)
                            OLpr=cat(1,OLpr,dif_pr);
                        end
                    end
                end
                if t2==0
                    cp1=cat(2,cp1,cp01(:,i));
                    pick(t1)=i;t1=t1+1;
                    t=t+1;
                end
            end
            cp2=cp02(:,pick(pick~=0));
        end
    end
    %************************如果无法再次划分，则对应节点子集确认为一个社区  
    if isempty(cp1)
        commu_n(commux)=subx;
        commux=commux+1;
        
    else%********************否则对节点子集做进一步划分
        OLm=[];%划分过程中将会产生的所有overlapping
        for i=1:size(cp1,2)
            dif_a=Diff(cp1(:,i),cp2(:,i),M1,M2,OL);
            if length(dif_a)<=para3&&~isempty(dif_a)
                OLm=cat(1,OLm,dif_a);
            end
        end
        %*********************sub1_v中进一步划分后产生的节点子集保存到后续列中
        ss1=liantong(s1,M1,cp1(M1+2,:));
        comm=unique(ss1);
        numb=nnz(comm);
        for i=1:numb
            subnum=N1(ss1==comm(i));
            sub1_v(end,suby+i)=length(subnum);
            sub1_v(1:length(subnum),suby+i)=subnum;
        end
        %*********************sub2_v中的节点子集进一步划分后产生的节点子集subm_v
        ss2=liantong(s2,M2,cp2(M2+2,:));
        comm=unique(ss2);
        numb=nnz(comm);
        subm_v=zeros(M2,numb);
        subm_n=zeros(1,numb);
        for i=1:numb
            subnum=N2(ss2==comm(i));
            subm_n(i)=length(subnum);
            subm_v(1:subm_n(i),i)=subnum;
        end
        %**************************将subm_v中的节点子集保存到sub2_v中，并与sub1_v相对应
        for i=(suby+1):(suby+numb)
            for j=1:numb
                dif_a=[setdiff(sub1_v(1:sub1_v(end,i),i),subm_v(1:subm_n(j),j));setdiff(subm_v(1:subm_n(j),j),sub1_v(1:sub1_v(end,i),i))];
                ol_n=length(dif_a);
                dif_b=setdiff(dif_a,[OL;OLm]);
                if isempty(dif_b)
                    sub2_v(1:M2,i)=subm_v(:,j);
                    sub2_v(end,i)=subm_n(j);
                    if isempty(dif_a)
                        dif_a=0;
                    end
                    subol_v(1:ol_n,i)=dif_a;
                    subol_v(end,i)=ol_n;
                    break
                end
            end
        end
        suby=suby+numb;
    end
    subx=subx+1;
end
%*****************************************整合最终的社区和overlapping
subol_v(end,:)=[];
OL=subol_v(subol_v~=0);
OL=unique(cat(1,unique(OL),unique(OLpr)));
commu_n=commu_n(commu_n~=0);
commu_v=sub1_v(:,commu_n);