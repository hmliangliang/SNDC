function [ J,R,F,MSE ] = SNDC(dataset,winsize)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%LE�㷨�Ĳο����̣�https://blog.csdn.net/qrlhl/article/details/78066994
%d����ά�������ά��  winsize:�������ڵĴ�С
tic;
t0=1;%��ʼ��ʱ���
sequence=0;%���δ��������Ư�����ݿ�����
p0=[];%��ʾ��һ�����ݿ�ľ������ĵ�
c0=[];%��ʾ��һ�����ݿ�ľ�������
J=[];
R=[];
F=[];
MSE=[];
N_max=15;
tau=0.5;
lama=2.3;
beta=0.001;
eta=3;
d=round(size(dataset,2)/2);
for i=1:size(dataset,1)
    %Lambda=[Lambda,exp(-lama*(i-t0))];%����ʵ����Ȩֵ
    if mod(i,winsize)==0%���û������ڻ���
        sequence=sequence+1;%��ǵ���ż�1
        data=dataset(i-winsize+1:i,:);%��ȡ����
        target=dataset(i-winsize+1:i,size(dataset,2));%��ȡʵ�����ǩ���������ԶԱ�����
        Lambda=zeros(1,size(data,1));
        for x=1:size(data,1)%�������ʵ����Ȩֵ
             Lambda(x)=exp(-lama*(i-winsize));
        end
        [data,n] = Partition( data );%�������Լ���ǰn�������Ƿ���������
        k=size(Roughset_partition(data(:,1:n)),2);%��ȡ�������ĵ���Ŀ
        [Idx,C,sumD]=kmeans(data(:,n+1:size(data,2)),k);%ִ��k-means�㷨
        delta=sum(sumD)/(size(data,1));%��ȡ�������ֵ
        data=Numeric(data,n);%ִ����ֵ������
        data=LE(data,d,delta);%�����ݽ���������˹����ӳ�併ά
        %��ά���������������
        [Idx,C,sumD]=kmeans(data,k);%ִ��k-means�㷨
        delta=sum(sumD)/(size(data,1));%��ȡ��ά�����ݼ��������ֵ
        Ne=Neighbourhood(data,delta);%��ȡÿһ�����������
        H=N_Entropy(Ne);%����ÿ�������������
        if sequence==1%��һ�ξ���
            p=Select_points(data,k,Lambda,Ne,beta);%���������ܶ�ѡ���ʼ�����ĵ�
            dis=Distance(data,p,Ne,H,delta);
            [value,cluster]=min(dis,[],2);%��ȡ����Ľ��
             %�����������
            c=zeros(1,k);
            for j=1:k
                c(j)=length(find(cluster==j));
            end
            %p,c��cluster����˵�ǰ���ݿ����Ľ��,��c��p��Ϊ��һ�����ݿ�ľ������
            c0=c;
            p0=p;
        else
           %����p0��c0ȷ��ÿ������Ĺ���
           dis=Distance(data,p0,Ne,H,delta);
           [value,cluster]=min(dis,[],2);%��ȡ����Ľ��
            p=update_center(data,Lambda,cluster,p0,N_max,eta);
            %�����������
            c=zeros(1,k);
            for j=1:k
                c(j)=length(find(cluster==j));
            end
            if length(c0)~=length(c)||acos(dot(sort(c),sort(c0))/(norm(c,2)*norm(c0,2)))>=tau%˵�������˸���Ư��
               p=Select_points(data,k,Lambda,Ne,beta);%����ѡ��������ĵ�
               dis=Distance(data,p,Ne,H,delta);
               [value,cluster]=min(dis,[],2);%��ȡ����Ľ��
               c=zeros(1,k);
               for j=1:k
                  c(j)=length(find(cluster==j));
               end
            else
               %dis=Distance(data,p,Ne,H,delta);
               dis=zeros(size(data,1),size(p,1));
               for xx=1:size(data,1)
                   for yy=1:size(p,1)
                       dis(xx,yy)=norm(data(xx,:)-p(yy,:),2);
                   end
               end
               [value,cluster]=min(dis,[],2);%��ȡ����Ľ��
               c=zeros(1,k);
               for j=1:k
                  c(j)=length(find(cluster==j));
               end
            end
            c0=c;
            p0=p;
        end
        %���������������ָ��
        [Jaccard,Rand,FM,mse] = cluster_performance(data,cluster,target,p);
        J=[J,Jaccard];
        R=[R,Rand];
        F=[F,FM];
        MSE=[MSE,mse];
        disp(['��',num2str(sequence),'�����ݿ����Ľ��Ϊ:Jaccard=',num2str(Jaccard),'  Rand=',num2str(Rand),' FM=',num2str(FM), '  mse=',num2str(mse)]);
        %��ǰ���ݿ鴦�����
        Lambda=[];%���Ȩֵ
        t0=i+1;%����ʱ��
    end
end
Jstd=std(J);
Rstd=std(R);
Fstd=std(F);
MSEstd=std(MSE);
% disp('J=');
% J'
% disp('R=');
% R'
% disp('F=');
% F'
% disp('MSE=');
% MSE'
J=mean(J);
R=mean(R);
F=mean(F);
MSE=mean(MSE);
disp(['�����ƽ�����Ϊ:ƽ��Jaccard=',num2str(J),'  ƽ��Rand=',num2str(R),'  ƽ��FM=',num2str(F), '  ƽ��mse=',num2str(MSE)]);
disp(['�����ƽ�����Ϊ:Jaccard����',num2str(Jstd),'  R�ķ���=',num2str(Rstd),'  FM����=',num2str(Fstd), '  ƽ��mse����=',num2str(MSEstd)]);
toc;
end

%�����ݶ��½���������¾�������
function p=update_center(data,Lambda,cluster,p,N_max,eta)
%dataΪ���ݼ�n*d,  clusterΪ�������ر��,  pΪ��ʼ�ľ������ĵ�,   N_max���ĵ�������,  eta�ݶ��½����Ĳ���
%LambdaΪʵ����Ȩֵ    eΪ���ĵ��ƶ�����С������ֵ
%mse=inf;%��¼���ĵ��ƶ��ľ���,�������ƶ�����ĳ�����ֵe����ȷ��,����ʵ����û������e,��������һ��С��N_max�������㷨�������ٶ�
N=1;
while N<=N_max%�������µ���ĿС��N_max
    N=N+1;
    for i=1:size(cluster,1)%����ÿһ�����ĵ�
        p(cluster(i,1),:)=p(cluster(i,1),:)+1/size(data,1)*eta*Lambda(i)*(data(i,:)-p(cluster(i,1),:));
    end
    mydist=zeros(size(data,1),size(p,1));%��ʼ���������Ϊ0
    for i=1:size(data,1)%�������ĵ�󣬶���ÿһ������
        for j=1:size(p,1)%���㵽���ĵ�֮���ŷʽ����
            mydist(i,j)=norm(data(i,:)-p(j,:),2);
        end
    end
    [value,cluster]=min(mydist,[],2);%������յľ�����ر��
end
end


%����data��ÿ��������ÿ�����ĵ�֮��������ؾ���
function dis=Distance(data,p,Ne,H,delta)
%dataΪ���ݼ�n*d,pΪ�������ĵ�k*d,NeΪdata��ÿ��ʵ��������n*n,disΪ����ľ���n*k
k=size(data,1);
pN=zeros(k,size(data,1));%pNΪ�������ĵ�p�����ݽ������򻮷ֵĽ������ʼ��pN
for x=1:size(p,1)%ÿ�����ĵ������
    for y=1:size(data,1)
        if norm(p(x,:)-data(y,:),2)<=delta
           pN(x,y)=1;
         end
    end
end
H_pN=N_Entropy(pN);%����ÿ�����ĵ��������
dis=zeros(size(data,1),k);%��ʼ��ÿ���������ĵ�֮��ľ������n*k
for x=1:size(data,1)%����ÿ���������ĵ�֮��ľ���
    for y=1:k
        dis(x,y)=abs(1/(length(find(Ne(x,:)==1)))*H(x,1)-1/(length(find(pN(x,:)==1)))*H_pN(x,1));
    end
end
end

%ѡ���k������ĳ�ʼ���ĵ�
function p=Select_points(data,k,Lambda,Ne,beta)
%kΪk-means�㷨�ĳ�ʼ���ĵ����Ŀ,LambdaΪʵ����Ȩֵ1*n,NeΪ�������������,betaԤ����Ĳ���,pΪѡ��������ĵ����k*d
p=[];%��ʼ������Ϊ��
density=zeros(1,size(data,1));%��ʼ������ʵ�����ܶ�ֵ
for i=1:size(data,1)%����ÿһ��ʵ�����ܶ�
   density(i)=Lambda(i)*(length(find(Ne(i,:)==1))/size(data,1));
end
[c,num]=max(density);%ѡ����ܶ����ĵ���Ϊһ���������ĵ�
pro=zeros(1,size(data,1));
pro=pro+density;
p=[p;data(num,:)];%��õ�һ���������ĵ�
pro(num)=[];
data(num,:)=[];
while size(p,1)<k %����������ĵ�û�дﵽk��ʱ
%����ÿһ��ʵ����Ϊ�������ĵĸ���
for i=1:size(data,1)
    for j=1:size(p,1)
        pro(i)=pro(i)+beta*norm(data(i,:)-p(j,:),2);%����ÿ�������Ϊ�������ĵ�ĸ���ֵ
    end
end
[c,num]=max(pro);%ѡ�����ֵ���ĵ���Ϊ�������ĵ�
p=[p;data(num,:)];%����һ����뵽������
pro(num)=[];
data(num,:)=[];
end
end

%����ÿ�������������
function H=N_Entropy(Ne)
%NeΪÿ�����������,deltaΪ����İ뾶��ֵ,HΪÿ�������������,����H�Ĵ�СΪn*1
N=Ne;%��ȡÿ�����������
H=zeros(size(N,2),1);%��ʼ����ֵ����
for i=1:size(N,1)%����ÿһ������
    for j=1:size(N,2)
        if N(i,j)==1%����j���ڶ���i��������
            H(i)=H(i)-(length(find(N(i,:)==1))/(size(N,2)))*log2((length(find(N(i,:)==1))/(size(N,2)))*1/(size(N,2)-length(find(N(i,:)==1))));
        end
    end
end
end


%ִ��LE(������˹����ӳ��)�㷨��ά
function data=LE(data,d,delta)
%dΪ��ά�������ά��,deltaΪ����İ뾶
neighbor=Neighbourhood(data,delta);%���ÿ�����������
W=zeros(size(data,1),size(data,1));%��ʼ��Ȩֵ
for x=size(data,1)%�������֮��������ȣ�������֮���Ȩֵ
   for y=1:size(data,1)
       W(x,y)=W(x,y)+length(intersect(find(neighbor(x,:)==1),find(neighbor(y,:)==1)))/length(find(neighbor(x,:)==1));
       W(y,x)=W(y,x)+length(intersect(find(neighbor(x,:)==1),find(neighbor(y,:)==1)))/length(find(neighbor(x,:)==1));
   end
end

for x=size(data,1)%ȨֵΪ0��ʾ����֮��ľ��������
   for y=x+1:size(data,1)
       if W(x,y)==0
           W(x,y)=inf;
           W(y,x)=inf;
       end
   end
end    
for x=size(data,1)%���������֮���
   for y=x+1:size(data,1)
        if W(x,y)==0
           W(x,y)=W(x,y)+1-Dijk(W,x,y);
           W(y,x)=W(y,x)+1-Dijk(W,x,y);
        end
   end
end
for x=size(data,1)%ȨֵΪ0��ʾ����֮��ľ��������,��ʱ�����ƶ�ӦΪ0
   for y=x+1:size(data,1)
       if W(x,y)<=0
           W(x,y)=0;
           W(y,x)=0;
       end
   end
end    
%���ɶԽǾ���D
D=zeros(1,size(W,2));%��ʼ��Ϊ�վ���
for x=1:size(W,2)
  D(x)=sum(W(:,x));
end
D=diag(D);
L=D-W;%����������˹����
[v,value]=eig(L);%������������ֵ����������
eigenvalue=diag(value)';%��ȡ���������ֵ
[q,ind]=sort(eigenvalue);%������ֵ����С�����������
ind=ind(2:(d+1));%��һ����С������ֵΪ0,��ȡd����С������ֵ��Ӧ�������������
data=v(:,ind);%��ȡ���յ���������,�õ���ά
end

%���ÿһ�����������
function neighbor=Neighbourhood(data,delta)
neighbor=zeros(size(data,1),size(data,1));%��ʼ���������
for i=size(data,1)
    for j=1:size(data,1)
        if norm(data(i,:)-data(j,:),2)<=delta %��j�������ڵ�i�������������
           neighbor(i,j)=1;
        end
    end
end
end

%������ֵ�����������������
function [data,n] = Partition(data)
N=20;%���������Ե�����ֵ������Ŀ
Attr_c=[];
Attr_n=[];
for i=1:size(data,2)
    if length(unique(data(:,i)))>N%˵��Ϊ��ֵ������
        Attr_n=[Attr_n,data(:,i)];
    else%˵��Ϊ����������
        Attr_c=[Attr_c,data(:,i)];
    end
end
data=[Attr_c,Attr_n];
n=size(Attr_c,2);%������������Ե���Ŀ
end

%����ɢ����ת��Ϊ��ֵ����
function data = Numeric(data,n)
Attr_c=data(:,1:n);%��ȡ����������
Attr_n=data(:,n+1:size(data,2));%��ȡ��ֵ������
Atrr=zeros(size(data,1),n);%��ʼ�����������Ե�����ֵ
Attr_n=zscore(Attr_n);%����ֵ���Խ��б�׼������
for i=1:n%����ÿ�����ԵĻ���
    for j=1:size(data,1)
        same_distance=0;%���浱ǰ������ȡ��ͬ����ֵ����֮���������
        diff_distance=1000000000;%���浱ǰ������ȡ��ͬ����ֵ����֮�����С����
        for x=1:size(data,1)
            dis=norm(Attr_n(j,:)-Attr_n(x,:),2);%����
            if Attr_c(j,i)==Attr_c(x,i)%�����������������ȡֵ��ͬ
                if same_distance<dis%�������ֵ
                    same_distance=dis;
                end
            else%�����������������ȡֵ����ͬ
                if diff_distance>dis%������Сֵ
                    diff_distance=dis;
                end
            end
        end
        Atrr(j,i)=diff_distance+same_distance;
    end     
 end
data=[Atrr,Attr_n];
end


function result = Roughset_partition(Attr)%�����Լ�Attr�����ݼ�data�Ļ���,Attr��һ��n*d�ķ����;���,result��һ�����ֽ��,��һ��1�ж��е�cell������
result=[];
for i=1:size(Attr,2)
    a=cell(1,length(unique(Attr(:,i))));%�γɳ�ʼ�Ļ���
    value=unique(Attr(:,i));%��������i�Ĳ�ͬ����ֵ
    for j=1:length(value)
        a{1,j}=find(Attr(:,i)==value(j,1));%ȡ�û��ֵĽ��
    end
    if i==1
        result=a;
    else%��ȡ�������Լ��Ļ��ֽ��
        tempset=cell(1,1);%������ʱ�Ľ��
        for m=1:size(result,2)%������ϸ����Ի��ּ��ϵĽ���
            for n=1:size(a,2)
                if (m==1)&&(n==1)
                   tempset{1,1}=intersect(result{m},a{n});
                else
                   tempset=[tempset,{intersect(result{m},a{n})}];
                end
            end
        end
        %�����ظ��ļ���
        label=zeros(1,size(tempset,2));
        for m=1:size(tempset,2)-1%�����ظ��Ľ��
            for n=m+1:size(tempset,2)
                if isequal(tempset{1,m},tempset{1,n})==1
                   label(n)=1;%������n������Ϊ�ظ��ģ���Ҫɾ��
                end
                if isempty(tempset{1,m})==1
                   label(m)=1;%������n������Ϊ�ظ��ģ���Ҫɾ��
                end
                if isempty(tempset{1,n})==1
                   label(n)=1;%������n������Ϊ�ظ��ģ���Ҫɾ��
                end
            end
        end
        label=find(label==1);%���Ҫɾ�����ϵ����
        tempset(:,label)=[];%ɾ���ظ��ļ���
        result=tempset;
        clear tempset;
    end
end
end

function distance = Dijk(W,st,e)
%DIJK Summary of this function goes here  
%W:Ȩֵ����   st:���������   e:�������յ�  
%pathΪ��̾����·��
n=length(W);%�ڵ���  
D = W(st,:);  
visit= ones(1:n); visit(st)=0;  
parent = zeros(1,n);%��¼ÿ���ڵ����һ���ڵ�   
path =[];    
for i=1:n-1  
    temp = [];  
    %��������������̾������һ���㣬ÿ�β����ظ�ԭ���Ĺ켣������visit�жϽڵ��Ƿ����  
    for j=1:n  
       if visit(j)  
           temp =[temp D(j)];  
       else  
           temp =[temp inf];  
       end  
    end    
    [value,index] = min(temp);  
    visit(index) = 0;  
    %���� �������index�ڵ㣬����㵽ÿ���ڵ��·�����ȸ�С������£���¼ǰ���ڵ㣬����������ѭ��  
    for k=1:n  
        if D(k)>D(index)+W(index,k)  
           D(k) = D(index)+W(index,k);  
           parent(k) = index;  
        end  
    end   
end  
distance = D(e);%��̾���  
%���ݷ�  ��β����ǰѰ������·��  
t = e;  
while t~=st && t>0  
 path =[t,path];  
  p=parent(t);t=p;  
end  
path =[st,path];%���·��   
end  


