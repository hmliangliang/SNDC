function [ J,R,F,MSE ] = SNDC(dataset,winsize)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%LE算法的参考过程：https://blog.csdn.net/qrlhl/article/details/78066994
%d：降维后的数据维度  winsize:滑动窗口的大小
tic;
t0=1;%开始的时间差
sequence=0;%标记未发生概念漂移数据块的序号
p0=[];%表示上一个数据块的聚类中心点
c0=[];%表示上一个数据块的聚类向量
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
    %Lambda=[Lambda,exp(-lama*(i-t0))];%计算实例的权值
    if mod(i,winsize)==0%采用滑动窗口机制
        sequence=sequence+1;%标记的序号加1
        data=dataset(i-winsize+1:i,:);%获取数据
        target=dataset(i-winsize+1:i,size(dataset,2));%获取实际类标签，用来测试对比性能
        Lambda=zeros(1,size(data,1));
        for x=1:size(data,1)%计算各个实例的权值
             Lambda(x)=exp(-lama*(i-winsize));
        end
        [data,n] = Partition( data );%划分属性集，前n个属性是分类型属性
        k=size(Roughset_partition(data(:,1:n)),2);%获取聚类中心的数目
        [Idx,C,sumD]=kmeans(data(:,n+1:size(data,2)),k);%执行k-means算法
        delta=sum(sumD)/(size(data,1));%获取邻域的阈值
        data=Numeric(data,n);%执行数值化操作
        data=LE(data,d,delta);%对数据进行拉普拉斯特征映射降维
        %降维后重新生成邻域的
        [Idx,C,sumD]=kmeans(data,k);%执行k-means算法
        delta=sum(sumD)/(size(data,1));%获取降维后数据集邻域的阈值
        Ne=Neighbourhood(data,delta);%获取每一个对象的邻域
        H=N_Entropy(Ne);%计算每个对象的邻域熵
        if sequence==1%第一次聚类
            p=Select_points(data,k,Lambda,Ne,beta);%依据邻域密度选择初始的中心点
            dis=Distance(data,p,Ne,H,delta);
            [value,cluster]=min(dis,[],2);%获取聚类的结果
             %计算聚类向量
            c=zeros(1,k);
            for j=1:k
                c(j)=length(find(cluster==j));
            end
            %p,c和cluster组成了当前数据块聚类的结果,把c与p作为下一个数据块的聚类参数
            c0=c;
            p0=p;
        else
           %根据p0与c0确定每个对象的归属
           dis=Distance(data,p0,Ne,H,delta);
           [value,cluster]=min(dis,[],2);%获取聚类的结果
            p=update_center(data,Lambda,cluster,p0,N_max,eta);
            %计算聚类向量
            c=zeros(1,k);
            for j=1:k
                c(j)=length(find(cluster==j));
            end
            if length(c0)~=length(c)||acos(dot(sort(c),sort(c0))/(norm(c,2)*norm(c0,2)))>=tau%说明发生了概念漂移
               p=Select_points(data,k,Lambda,Ne,beta);%重新选择聚类中心点
               dis=Distance(data,p,Ne,H,delta);
               [value,cluster]=min(dis,[],2);%获取聚类的结果
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
               [value,cluster]=min(dis,[],2);%获取聚类的结果
               c=zeros(1,k);
               for j=1:k
                  c(j)=length(find(cluster==j));
               end
            end
            c0=c;
            p0=p;
        end
        %计算聚类结果的评价指标
        [Jaccard,Rand,FM,mse] = cluster_performance(data,cluster,target,p);
        J=[J,Jaccard];
        R=[R,Rand];
        F=[F,FM];
        MSE=[MSE,mse];
        disp(['第',num2str(sequence),'个数据块聚类的结果为:Jaccard=',num2str(Jaccard),'  Rand=',num2str(Rand),' FM=',num2str(FM), '  mse=',num2str(mse)]);
        %当前数据块处理完成
        Lambda=[];%清空权值
        t0=i+1;%更新时间
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
disp(['聚类的平均结果为:平均Jaccard=',num2str(J),'  平均Rand=',num2str(R),'  平均FM=',num2str(F), '  平均mse=',num2str(MSE)]);
disp(['聚类的平均结果为:Jaccard方差',num2str(Jstd),'  R的方差=',num2str(Rstd),'  FM方差=',num2str(Fstd), '  平均mse方差=',num2str(MSEstd)]);
toc;
end

%采用梯度下降法方向更新聚类中心
function p=update_center(data,Lambda,cluster,p,N_max,eta)
%data为数据集n*d,  cluster为聚类的类簇标号,  p为初始的聚类中心点,   N_max最大的迭代数次,  eta梯度下降法的步长
%Lambda为实例的权值    e为中心点移动的最小距离阈值
%mse=inf;%记录中心点移动的距离,由于其移动距离的长度阈值e不好确定,故在实验中没有设置e,而是设置一个小的N_max来控制算法的收敛速度
N=1;
while N<=N_max%迭代更新的数目小于N_max
    N=N+1;
    for i=1:size(cluster,1)%更新每一个中心点
        p(cluster(i,1),:)=p(cluster(i,1),:)+1/size(data,1)*eta*Lambda(i)*(data(i,:)-p(cluster(i,1),:));
    end
    mydist=zeros(size(data,1),size(p,1));%初始化距离矩阵为0
    for i=1:size(data,1)%更新中心点后，对于每一个样本
        for j=1:size(p,1)%计算到中心点之间的欧式距离
            mydist(i,j)=norm(data(i,:)-p(j,:),2);
        end
    end
    [value,cluster]=min(mydist,[],2);%获得最终的聚类类簇标号
end
end


%计算data中每个对象与每个中心点之间的邻域熵距离
function dis=Distance(data,p,Ne,H,delta)
%data为数据集n*d,p为聚类中心点k*d,Ne为data中每个实例的邻域n*n,dis为计算的距离n*k
k=size(data,1);
pN=zeros(k,size(data,1));%pN为根据中心点p对数据进行邻域划分的结果，初始化pN
for x=1:size(p,1)%每个中心点的邻域
    for y=1:size(data,1)
        if norm(p(x,:)-data(y,:),2)<=delta
           pN(x,y)=1;
         end
    end
end
H_pN=N_Entropy(pN);%计算每个中心点的邻域熵
dis=zeros(size(data,1),k);%初始化每个对象到中心点之间的距离矩阵n*k
for x=1:size(data,1)%计算每个对象到中心点之间的距离
    for y=1:k
        dis(x,y)=abs(1/(length(find(Ne(x,:)==1)))*H(x,1)-1/(length(find(pN(x,:)==1)))*H_pN(x,1));
    end
end
end

%选择出k个聚类的初始中心点
function p=Select_points(data,k,Lambda,Ne,beta)
%k为k-means算法的初始中心点的数目,Lambda为实例的权值1*n,Ne为各个对象的邻域,beta预定义的参数,p为选择出的中心点矩阵k*d
p=[];%初始化矩阵为空
density=zeros(1,size(data,1));%初始化各个实例的密度值
for i=1:size(data,1)%计算每一个实例的密度
   density(i)=Lambda(i)*(length(find(Ne(i,:)==1))/size(data,1));
end
[c,num]=max(density);%选择出密度最大的点作为一个聚类中心点
pro=zeros(1,size(data,1));
pro=pro+density;
p=[p;data(num,:)];%获得第一个聚类中心点
pro(num)=[];
data(num,:)=[];
while size(p,1)<k %当聚类的中心点没有达到k个时
%计算每一个实例称为聚类中心的概率
for i=1:size(data,1)
    for j=1:size(p,1)
        pro(i)=pro(i)+beta*norm(data(i,:)-p(j,:),2);%计算每个对象成为聚类中心点的概率值
    end
end
[c,num]=max(pro);%选择概率值最大的点作为聚类中心点
p=[p;data(num,:)];%把这一点加入到集合中
pro(num)=[];
data(num,:)=[];
end
end

%计算每个对象的邻域熵
function H=N_Entropy(Ne)
%Ne为每个对象的邻域,delta为邻域的半径阈值,H为每个对象的邻域熵,矩阵H的大小为n*1
N=Ne;%获取每个对象的邻域
H=zeros(size(N,2),1);%初始化熵值矩阵
for i=1:size(N,1)%对于每一个对象
    for j=1:size(N,2)
        if N(i,j)==1%对象j属于对象i的邻域内
            H(i)=H(i)-(length(find(N(i,:)==1))/(size(N,2)))*log2((length(find(N(i,:)==1))/(size(N,2)))*1/(size(N,2)-length(find(N(i,:)==1))));
        end
    end
end
end


%执行LE(拉普拉斯特征映射)算法降维
function data=LE(data,d,delta)
%d为降维后的数据维数,delta为邻域的半径
neighbor=Neighbourhood(data,delta);%求解每个对象的邻域
W=zeros(size(data,1),size(data,1));%初始化权值
for x=size(data,1)%计算对象之间的隶属度，求解对象之间的权值
   for y=1:size(data,1)
       W(x,y)=W(x,y)+length(intersect(find(neighbor(x,:)==1),find(neighbor(y,:)==1)))/length(find(neighbor(x,:)==1));
       W(y,x)=W(y,x)+length(intersect(find(neighbor(x,:)==1),find(neighbor(y,:)==1)))/length(find(neighbor(x,:)==1));
   end
end

for x=size(data,1)%权值为0表示他们之间的距离无穷大
   for y=x+1:size(data,1)
       if W(x,y)==0
           W(x,y)=inf;
           W(y,x)=inf;
       end
   end
end    
for x=size(data,1)%非邻域对象之间的
   for y=x+1:size(data,1)
        if W(x,y)==0
           W(x,y)=W(x,y)+1-Dijk(W,x,y);
           W(y,x)=W(y,x)+1-Dijk(W,x,y);
        end
   end
end
for x=size(data,1)%权值为0表示他们之间的距离无穷大,此时的相似度应为0
   for y=x+1:size(data,1)
       if W(x,y)<=0
           W(x,y)=0;
           W(y,x)=0;
       end
   end
end    
%生成对角矩阵D
D=zeros(1,size(W,2));%初始化为空矩阵
for x=1:size(W,2)
  D(x)=sum(W(:,x));
end
D=diag(D);
L=D-W;%生成拉普拉斯矩阵
[v,value]=eig(L);%计算矩阵的特征值与特征向量
eigenvalue=diag(value)';%提取矩阵的特征值
[q,ind]=sort(eigenvalue);%对特征值按由小到大进行排序
ind=ind(2:(d+1));%第一个最小的特征值为0,获取d个最小的特征值对应的特征向量序号
data=v(:,ind);%获取最终的特征向量,得到降维
end

%求解每一个对象的邻域
function neighbor=Neighbourhood(data,delta)
neighbor=zeros(size(data,1),size(data,1));%初始化邻域矩阵
for i=size(data,1)
    for j=1:size(data,1)
        if norm(data(i,:)-data(j,:),2)<=delta %第j个对象处于第i个对象的邻域内
           neighbor(i,j)=1;
        end
    end
end
end

%划分数值型属性与分类型属性
function [data,n] = Partition(data)
N=20;%分类型属性的属性值最大的数目
Attr_c=[];
Attr_n=[];
for i=1:size(data,2)
    if length(unique(data(:,i)))>N%说明为数值型属性
        Attr_n=[Attr_n,data(:,i)];
    else%说明为分类型属性
        Attr_c=[Attr_c,data(:,i)];
    end
end
data=[Attr_c,Attr_n];
n=size(Attr_c,2);%计算分类型属性的数目
end

%将离散属性转换为数值属性
function data = Numeric(data,n)
Attr_c=data(:,1:n);%获取分类型属性
Attr_n=data(:,n+1:size(data,2));%获取数值型属性
Atrr=zeros(size(data,1),n);%初始化分类型属性的属性值
Attr_n=zscore(Attr_n);%对数值属性进行标准化操作
for i=1:n%计算每个属性的划分
    for j=1:size(data,1)
        same_distance=0;%保存当前对象与取相同属性值对象之间的最大距离
        diff_distance=1000000000;%保存当前对象与取不同属性值对象之间的最小距离
        for x=1:size(data,1)
            dis=norm(Attr_n(j,:)-Attr_n(x,:),2);%距离
            if Attr_c(j,i)==Attr_c(x,i)%两个对象分类型属性取值相同
                if same_distance<dis%更新最大值
                    same_distance=dis;
                end
            else%两个对象分类型属性取值不相同
                if diff_distance>dis%更新最小值
                    diff_distance=dis;
                end
            end
        end
        Atrr(j,i)=diff_distance+same_distance;
    end     
 end
data=[Atrr,Attr_n];
end


function result = Roughset_partition(Attr)%求属性集Attr对数据集data的划分,Attr是一个n*d的分类型矩阵,result是一个划分结果,是一个1行多列的cell型数组
result=[];
for i=1:size(Attr,2)
    a=cell(1,length(unique(Attr(:,i))));%形成初始的划分
    value=unique(Attr(:,i));%保存属性i的不同属性值
    for j=1:length(value)
        a{1,j}=find(Attr(:,i)==value(j,1));%取得划分的结果
    end
    if i==1
        result=a;
    else%获取整个属性集的划分结果
        tempset=cell(1,1);%保存临时的结果
        for m=1:size(result,2)%求解与上个属性划分集合的交集
            for n=1:size(a,2)
                if (m==1)&&(n==1)
                   tempset{1,1}=intersect(result{m},a{n});
                else
                   tempset=[tempset,{intersect(result{m},a{n})}];
                end
            end
        end
        %消除重复的集合
        label=zeros(1,size(tempset,2));
        for m=1:size(tempset,2)-1%查找重复的结合
            for n=m+1:size(tempset,2)
                if isequal(tempset{1,m},tempset{1,n})==1
                   label(n)=1;%表明第n个集合为重复的，需要删除
                end
                if isempty(tempset{1,m})==1
                   label(m)=1;%表明第n个集合为重复的，需要删除
                end
                if isempty(tempset{1,n})==1
                   label(n)=1;%表明第n个集合为重复的，需要删除
                end
            end
        end
        label=find(label==1);%获得要删除集合的序号
        tempset(:,label)=[];%删除重复的集合
        result=tempset;
        clear tempset;
    end
end
end

function distance = Dijk(W,st,e)
%DIJK Summary of this function goes here  
%W:权值矩阵   st:搜索的起点   e:搜索的终点  
%path为最短距离的路径
n=length(W);%节点数  
D = W(st,:);  
visit= ones(1:n); visit(st)=0;  
parent = zeros(1,n);%记录每个节点的上一个节点   
path =[];    
for i=1:n-1  
    temp = [];  
    %从起点出发，找最短距离的下一个点，每次不会重复原来的轨迹，设置visit判断节点是否访问  
    for j=1:n  
       if visit(j)  
           temp =[temp D(j)];  
       else  
           temp =[temp inf];  
       end  
    end    
    [value,index] = min(temp);  
    visit(index) = 0;  
    %更新 如果经过index节点，从起点到每个节点的路径长度更小，则更新，记录前趋节点，方便后面回溯循迹  
    for k=1:n  
        if D(k)>D(index)+W(index,k)  
           D(k) = D(index)+W(index,k);  
           parent(k) = index;  
        end  
    end   
end  
distance = D(e);%最短距离  
%回溯法  从尾部往前寻找搜索路径  
t = e;  
while t~=st && t>0  
 path =[t,path];  
  p=parent(t);t=p;  
end  
path =[st,path];%最短路径   
end  


