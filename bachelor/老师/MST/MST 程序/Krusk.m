function st=Krusk(D)
n=length(D);
[D1,D2,D3]=find(D);
EDGE=uint16([D1 D2 D3]);
clear D D1 D2 D3;
EDGE=sortrows(EDGE,3);
st=zeros(n-1,3);
c=(1:n)';
num=ones(n,1);
edgenum=1;
edgesel=1;
while edgenum<n
    [a1,c]=Find(EDGE(edgesel,1),c);
    [a2,c]=Find(EDGE(edgesel,2),c);
    if a1~=a2
        st(edgenum,:)=double(EDGE(edgesel,:));
        edgenum=edgenum+1;
        if num(a1)<=num(a2)
            c(a1)=a2;
            num(a2)=num(a2)+num(a1);
        else
            c(a2)=a1;
            num(a1)=num(a1)+num(a2);
        end
    end
    edgesel=edgesel+1;
end
st=sparse(st(:,1),st(:,2),st(:,3),n,n);

function [y,d]=Find(b,c)
y=b;
while y~=c(y)
    y=c(y);
end
while y~=c(b)
    t=c(b);
    c(b)=y;
    b=t;
end
d=c;