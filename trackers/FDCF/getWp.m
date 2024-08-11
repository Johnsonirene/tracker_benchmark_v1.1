function [Wp,W,tzxl,tzz,L,Qnew]=getWp(W,kk,p,N)
n=N;
W=(W'+W)/2;
D=diag(sum(W,2));
L=D-W;
[tzxl,tzz]=eig(L);%返回特征值的矩阵tzz和对角矩阵tzxl
[~,index]=sort(diag(tzz));
Qold=tzxl(:,index(1:kk));
R=zeros(kk,kk);
G=zeros(n,kk);
ZJ=zeros(n,kk);
itermax=100;
iter=1;
e=1e-5;
while iter<itermax
    for i=1:kk
        qk(i)=norm(Qold(:,i),p)^p;
    end
    for i=1:n
        s=0;
        for j=1:kk
            col=find(W(i,:)~=0);
            s=W(i,col)*(abs(Qold(i,j)-Qold(col,j)).^(p-1).*sign(Qold(i,j)-Qold(col,j)));
            ZJ(i,j)=(s-abs(Qold(i,j))^(p-1)*sign(Qold(i,j))/qk(j))/qk(j);
        end
    end
    G=ZJ-Qold*ZJ'*Qold;
%     yita=0.01*norm(Qold,1)/norm(G,1);
    yita=0.01*sum(sum(abs(Qold)))/sum(sum(abs(G)));
    Qnew=Qold-yita*G;
%     chazhi=sum(sum(abs(Qnew-Qold)));
    chazhi=norm(Qnew-Qold,'fro');
    if iter>1&chazhi<e
        break
    end
     iter=iter+1;
     Qold=Qnew;
end
for i=1:kk
    s=0;
    for j=1:n
        col=find(W(j,:)~=0);
        s=s+W(j,col)*abs(Qnew(j,i)-Qnew(col,i)).^p;
    end
    qk_=norm(Qnew(:,i),p)^p;
    R(i,i)=s/qk_;
end
Lp=Qnew*R*Qnew';
Dp=diag(diag(Lp));
Wp=Dp-Lp;
end

