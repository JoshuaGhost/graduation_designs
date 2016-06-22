function c=liantong(x,m,k)
n=size(x,1);
c=(1:m)';
x(k,:)=0;
num=ones(m,1);
for i=1:n
    if any(x(i,:))
        [a1,c]=Find(x(i,1),c);
        [a2,c]=Find(x(i,2),c);
        if a1~=a2&&num(a1)<=num(a2)
            c(a1)=a2;
            num(a2)=num(a2)+num(a1);
        elseif a1~=a2&&num(a1)>num(a2)
                c(a2)=a1;
                num(a1)=num(a1)+num(a2);
        end
    end
end
for j=1:m
    [non,c]=Find(j,c);
end
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



