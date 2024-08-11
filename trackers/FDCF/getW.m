function [W]=getW(X,N)
% S=(X-mean(X,2))'*(X-mean(X,2))
[d,n]=size(X);%d代表维度，n代表有多少个样本
k=N-1;
% neighbor=zeros(n,k);
W=zeros(n,n);
for i=1:n
    dis=X-X(:,i);%减去第i列
    juli=sum(dis.^2,1); %计算其他样本到第i样本的距离
    [~,index]=sort(juli,2);%
%     neighbor(i,:)=index(2:k+1);
    for j=2:k+1
        W(i,index(j))=exp(-sqrt(sum((X(:,i)-X(:,index(j))).^2))/0.5);
    end
end
%%
W=W;
end