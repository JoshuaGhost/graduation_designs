function dif_a=Diff(p1,p2,M1,M2,OL)
dif_a=[setdiff(p1(1:p1(M1+1)),p2(1:p2(M2+1)));setdiff(p2(1:p2(M2+1)),p1(1:p1(M1+1)))];
dif_a=setdiff(dif_a,OL);
if length(dif_a)>length(p1)/2
    dif_b=[setdiff(p1(1:p1(M1+1)),p2(p2(M2+1)+1:M2));setdiff(p2(p2(M2+1)+1:M2),p1(1:p1(M1+1)))];
    dif_b=setdiff(dif_b,OL); 
    if length(dif_a)>length(dif_b)
        dif_a=dif_b;
    end
end