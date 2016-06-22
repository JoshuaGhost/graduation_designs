function D=distance(x,para1)
M=length(x);
D=zeros(M);
dpre=false(M);  %dpre��������Ҫ����ľ���
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
            t1=x(i,j);%�������A��·��
            %*****************t2Ϊ�������B��·������·������2���ߣ�1���м�ڵ�
            pp=x(:,i);               %�ڵ�i�������е��������
            pp(j)=0;                 %ȥ��·��A
            [a,none]=find(pp~=0);    %aΪ��iֱ�������Ľڵ㼯����i�������еĽӿ�
            [p,none]=find(x(a,j)~=0);%pΪa����jֱ�������Ľڵ���a�е�λ�ã�
            t2=length(p);
            %*****************t3Ϊ�������C��·������·������3���ߣ�2���м�ڵ�
            p1=a(p);     %p1Ϊ·��B�����������м�ڵ㣬��ͬʱ��Ϊi��j�Ľӿ�
            a(p)=[];     %ȥ��i��j�Ĺ��ýӿڣ�a��ֻʣ��·��A��B��δռ�õĽӿ�
            pp=x(:,j);   %�ڵ�j�������е��������
            pp([p1;i])=0;%ȥ��·��A�Լ�i��j�Ĺ��ýӿ�
            [b,none]=find(pp~=0);%b��ֻʣ��·��A��Bδռ�õĽӿ�
            G=x(a,b);       %G��ʾ��a��b��ʣ��Ľӿ�֮�����ͨ��������G(u,v)
            gi=any(G,2);    %Ϊ1�����ʾ·��i-u-v-j���ڣ����������C��gi=any(G,2)
            gj=any(G,1);    %��gj=any(G,1)�����Ƴ�i��jû�õĽӿڣ�Ϊ������ÿ���ӿ�    
            G=G(gi,gj);     %ֻ��ʹ��1�ε�������ѡ��min(n1,n2)���Ա�֤ÿ���ӿ�ֻʹ
            [n1,n2]=size(G);%��1��
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