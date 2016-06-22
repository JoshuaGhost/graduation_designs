function nmi=GETNMI(community,commu_v,M)
communityb=zeros(M,2);
t=0;
for i=1:size(commu_v,2)
    for j=1:commu_v(end,i)
    communityb(j+t,:)=[commu_v(j,i),i];
    end
    t=t+commu_v(end,i);
end
communityb=sortrows(communityb,1);
NMI=zeros(max(community(:,2)),size(commu_v,2));
for i=1:M
    NMI(community(i,2),communityb(i,2))=NMI(community(i,2),communityb(i,2))+1;
end
Ni=sum(NMI,2);
Nj=sum(NMI,1);
Na=0;
Nb1=0;
Nb2=0;
for i=1:size(NMI,1)
    Nb1=Nb1+Ni(i)*log(Ni(i)/M);
    for j=1:size(NMI,2)
        if NMI(i,j)~=0
        Na=Na+(-2*NMI(i,j)*log((NMI(i,j)*M)/(Ni(i)*Nj(j))));
        end
    end
end
for j=1:size(NMI,2)
Nb2=Nb2+Nj(j)*log(Nj(j)/M);
end
nmi=Na/(Nb1+Nb2);
