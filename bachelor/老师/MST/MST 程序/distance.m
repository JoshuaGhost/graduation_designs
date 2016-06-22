function D=distance(x,para1)
M=length(x);
D=zeros(M);
dpre=false(M);  %dpre给出了需要计算的距离
for i=1:M
    m1=find(x(:,i)~=0);
    m2=find(sum(x(:,m1),2)~=0);
    m3=find(sum(x(:,m2),2)~=0);
    m=[m1;m2;m3];
    m=unique(m);
    m=setdiff(m,1:i);
    dpre(i,m)=1;
end
for i=1:M
    for j=(i+1):M
        if dpre(i,j)
            t1=x(i,j);%符合情况A的路径
            %*****************t2为符合情况B的路径，即路径包含2条边，1个中间节点
            pp=x(:,i);               %节点i在网络中的连接情况
            pp(j)=0;                 %去除路径A
            [a,none]=find(pp~=0);    %a为与i直接相连的节点集，即i在网络中的接口
            [p,none]=find(x(a,j)~=0);%p为a中与j直接相连的节点在a中的位置，
            t2=length(p);
            %*****************t3为符合情况C的路径，即路径包含3条边，2个中间节点
            p1=a(p);     %p1为路径B包含的所有中间节点，即同时作为i和j的接口
            a(p)=[];     %去除i和j的共用接口，a中只剩下路径A和B均未占用的接口
            pp=x(:,j);   %节点j在网络中的连接情况
            pp([p1;i])=0;%去除路径A以及i和j的共用接口
            [b,none]=find(pp~=0);%b中只剩下路径A和B未占用的接口
            G=x(a,b);       %G表示了a和b中剩余的接口之间的连通情况，如果G(u,v)
            gi=any(G,2);    %为1，则表示路径i-u-v-j存在，即符合情况C。gi=any(G,2)
            gj=any(G,1);    %和gj=any(G,1)可以移除i和j没用的接口；为了满足每个接口    
            G=G(gi,gj);     %只能使用1次的条件，选择min(n1,n2)可以保证每个接口只使
            [n1,n2]=size(G);%用1次
            t3=min(n1,n2);  
            if (i == 2) && (j ==3)
                disp(n1);
                disp(n2);
                disp(t1);
                disp(' ');
                disp(t2);
                disp(' ');
                disp(t3);
            end
            d=t1+t2/2+t3/3;
            D(i,j)=1/d;
        else
            D(i,j)=10;
        end
    end
end
D=D.*(ones(M)-x*para1);
D=sparse(D);
D=tril(D+D');