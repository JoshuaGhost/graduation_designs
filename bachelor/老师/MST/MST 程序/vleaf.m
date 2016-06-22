function [leaf,leafnum]=vleaf(x)
leaf=zeros(length(x),2);
number=1;
col=sum(x);
while any(col==1)
    [l1,l2]=find(col==1);
    for i=1:length(l2)
        [lr,lc]=find(x(:,l2(i))==1);
        leaf(number,:)=[lr l2(i)];
        number=number+1;
        x(lr,l2(i))=0;
        x(l2(i),lr)=0;
    end 
    col=sum(x);
end
if any(any(leaf))
    leaf=leaf(1:number-1,:);
    for i=1:number-1
        leaf(leaf(:,1)==leaf(number-i,2))=leaf(number-i,1);
    end
    leaf=sortrows(leaf,2);
    branch=sortrows(unique(leaf(:,1)));
    leafnum=[branch zeros(length(branch),1)];
    for i=1:length(branch)
        leafnum(i,2)=length(find(leaf==branch(i)));
    end
    for i=1:number-1
        branch(branch>leaf(number-i,2))=branch(branch>leaf(number-i,2))-1;
    end
    leafnum(:,1)=branch;
else
    leaf=0;
    leafnum=0;
end
