function FUZZYMONODISKNN(database,type,K,k,upd,dod)
%number222=round(0.7*size(database,1)); %% size(array,1) returns the number of rows; size(array,2) returns the number of columns. 
                                       %%round(a) returns the nearest integer of a.
                                       %%database should be loaded first.
                                      
% train_database=database(1:number222,:); 
% test_database=database(number222+1:end,:);
% 70% of database are used to train, and 30% are used to test.
Total_Num=size(database,1);
Train_Num=round(0.7*Total_Num);
idx=randperm(Total_Num);
idx=idx(1:Train_Num);
train_database=database(idx,:);
test_database=database;
test_database(idx,:)=[]; 

N=size(train_database,1);%N是训练样例的个数
n=size(train_database,2)-1;%n是条件属性的个数
M=size(test_database,1);%M是待测样本的个数
LABLE=test_database(:,end);
train_attribute=train_database(:,1:n);
test_attribute=test_database(:,1:n);
Corr=zeros(1,n);
for i=1:n
    pre_Corr=corrcoef([database(:,i),database(:,end)]);
    Corr(i)=pre_Corr(2,1);
end
IN_MARK=find(Corr>0);%存储大于0的元素的位置，对应跟决策是单调递增关系的属性
disp('与决策属性之间存在单调递增关系的属性为：');
disp(IN_MARK); 
DE_MARK=find(Corr<0);%存储小于0的元素的位置，对应跟决策是单调递增关系的属性
disp('与决策属性之间存在单调递减关系的属性为：');
disp(DE_MARK);
IN_CORR=[];%存储大于0的元素，正相关的系数
IN_NUM=1;
DE_CORR=[];%存储小于0的元素，负相关的系数
DE_NUM=1;
for i=1:n
    if Corr(1,n)>0
        IN_CORR(IN_NUM)=Corr(1,n);
        IN_NUM=IN_NUM+1;
    end
    if Corr(1,n)<0
        DE_CORR(DE_NUM)=Corr(1,n);
        DE_NUM=DE_NUM+1;
    end
end

IN_TRUE_label=zeros(M,N);
if ~isempty(IN_MARK)
    for i=1:M
        for j=1:N
            if test_attribute(i,IN_MARK)>=train_attribute(j,IN_MARK)
                IN_TRUE_label(i,j)=1;
            end
        end
    end
else
    IN_TRUE_label=[];
end

DE_TRUE_label=zeros(M,N);
if ~isempty(DE_MARK)
    for i=1:M
        for j=1:N
            if test_attribute(i,DE_MARK)<=train_attribute(j,DE_MARK)
                DE_TRUE_label(i,j)=1;
            end
        end
    end
else
    DE_TRUE_label=[];
end

Upward=zeros(M,N);%存储待测样本大于训练样本的程度:每个条件属性的大于程度的加权平均值，权值为每个条件属性与决策属性之间的相关系数
Real_Upward=zeros(M,N);
Real_Downward=zeros(M,N);
for i=1:M
    for j=1:N
        up_pre_fuzzy_degree=zeros(n,1);
        for r=1:n
            if Corr(r)>0
                up_pre_fuzzy_degree(r,1)=1/(1+exp(-k*((test_attribute(i,r))-(train_attribute(j,r)))));
            end
            if Corr(r)<0
                up_pre_fuzzy_degree(r,1)=1/(1+exp(k*((test_attribute(i,r))-(train_attribute(j,r)))));
            end
        end
        Upward(i,j)=abs(Corr)*up_pre_fuzzy_degree;
        if ~isempty(IN_MARK)&&~isempty(DE_MARK)&&IN_TRUE_label(i,j)==1&&DE_TRUE_label(i,j)==1
            Real_Upward(i,j)=Upward(i,j);
        end
        if ~isempty(IN_MARK)&&isempty(DE_MARK)&&IN_TRUE_label(i,j)==1
            Real_Upward(i,j)=Upward(i,j);
        end
        if isempty(IN_MARK)&&~isempty(DE_MARK)&&DE_TRUE_label(i,j)==1
            Real_Upward(i,j)=Upward(i,j);
        end
        
        if ~isempty(IN_MARK)&&~isempty(DE_MARK)&&IN_TRUE_label(i,j)==0&&DE_TRUE_label(i,j)==0
            Real_Downward(i,j)=Upward(i,j);
        end
        if ~isempty(IN_MARK)&&isempty(DE_MARK)&&IN_TRUE_label(i,j)==0
            Real_Downward(i,j)=Upward(i,j);
        end
        if isempty(IN_MARK)&&~isempty(DE_MARK)&&DE_TRUE_label(i,j)==0
            Real_Downward(i,j)=Upward(i,j);
        end
%         up_pre_fuzzy_degree=zeros(n,1);
%         for r=1:n
%             up_pre_fuzzy_degree(r,1)=1/(1+exp(-k*((test_attribute(i,r))-(train_attribute(j,r)))));
%         end
%         Upward(i,j)=Corr*up_pre_fuzzy_degree;

    end
end


% Downward=zeros(M,N);%存储待测样本小于训练样本的程度
% for i=1:M
%     for j=1:N
%         down_pre_fuzzy_degree=zeros(n,1);
%         for r=1:n
%             down_pre_fuzzy_degree(r,1)=1/(1+exp(k*((test_attribute(i,r))-(train_attribute(j,r)))));
%         end
%         Downward(i,j)=abs(Corr)*down_pre_fuzzy_degree;
%     end
% end
UU=Upward';
UPWARD=mapminmax(UU',0,1);
LABLE_BELOW=zeros(M,1);
for i=1:M
    num=1;
    Pre_Lable_Below=[];
    for j=1:N
        if UPWARD(i,j)>=upd
            Pre_Lable_Below(num)=train_database(j,end);
            num=num+1;
        end
    end
    if num>1
        LABLE_BELOW(i)=max(Pre_Lable_Below);
    elseif num==1
        LABLE_BELOW(i)=-inf;
    end
end
LABLE_UP=zeros(M,1);
for i=1:M
    num=1;
    Pre_Lable_Up=[];
    for j=1:N
        if  UPWARD(i,j)<=dod
            Pre_Lable_Up(num)=train_database(j,end);
            num=num+1;
        end
%         if Downward(i,j)>=dod
%             Pre_Lable_Up(num)=train_database(j,end);
%             num=num+1;
%         end

    end
    if num>1
        LABLE_UP(i)=min(Pre_Lable_Up);
    elseif num==1
        LABLE_UP(i)=inf;
    end
end
Prediction=zeros(M,1);
REGRESSION=0;
CLASSIFIER=1;
Num=0;%Num表示周围没有训练样本的待测样本个数
Num_test=[];%存储周围没有训练样本的待测样本的角标
for i=1:M
    trainset=[];
    num=1;
    for j=1:N
        if train_database(j,end)>=LABLE_BELOW(i,1)&&train_database(j,end)<=LABLE_UP(i,1)
            trainset(num)=j;
            num=num+1;
        end
    end
    if num==1
%         disp('第');
%         disp(i);
%         disp('个待测样本周围没有待选训练样本，请调整程度参数或k值');
        Num=Num+1;
        Num_test(Num)=i;
        Prediction(i,1)=NaN;
    elseif num>1
        DIS=zeros(1,num-1);
        for k=1:num-1
            DIS(1,k)=sqrt(sum((test_attribute(i,:)-train_attribute(trainset(k),:)).^2));
        end
        if K>num-1
            disp('K should be less than or equal to');
            disp(num-1);
            return;
        else
            [SORTED,MARK]=sort(DIS,2);
            SORTED_DIS=SORTED(:,1:K);
            K_MARK=MARK(:,1:K);
            Pre_Prediction=zeros(1,K);
            for p=1:K
                Pre_Prediction(1,p)=train_database(K_MARK(1,p),end);
            end
            if type==REGRESSION
                if SORTED_DIS==0
                    Prediction(i)=Pre_Prediction(1,1);
                else
                    Prediction(i,1)=mean(Pre_Prediction,2);
                end
%                 [m,~] = size(SORTED_DIS);
%                 for r = 1:m
%                     SORTED_DIS(r,:)=SORTED_DIS(r,:)/norm(SORTED_DIS(r,:));
%                 end
%                 WEIGHT=1-SORTED_DIS;
%                 Prediction(i,1)=WEIGHT*(Pre_Prediction');
            end
            if type==CLASSIFIER
                b=[];
                [m,~]=size(Pre_Prediction);
                for s=1:m
                    [k,l]=mode(Pre_Prediction(s,:));
                    b=[b;k l];
                end
                Prediction(i,1)=b(1);
            end
        end
    end
end
% NMP=0;
% for i=1:M
%     for j=1:M
%         if test_attribute(i,:)>=test_attribute(j,:)&Prediction(i,1)<Prediction(j,1)
%             NMP=NMP+1;
%         end
%     end
% end
IN_TRUE=zeros(M,M);%逻辑值矩阵，判断待测样例i是否大于待测样例j
if ~isempty(IN_MARK)
    for i=1:M
        for j=1:M
            if test_attribute(i,IN_MARK)>=test_attribute(j,IN_MARK)
                IN_TRUE(i,j)=1;
            end
        end
    end
else
    IN_TRUE=[];
end

DE_TRUE=zeros(M,M);
if ~isempty(DE_MARK)
    for i=1:M
        for j=1:M
            if test_attribute(i,DE_MARK)<=test_attribute(j,DE_MARK)
                DE_TRUE(i,j)=1;
            end
        end
    end
else
    DE_TRUE=[];
end

DECISION_TRUE=zeros(M,M);
for i=1:M
    for j=1:M
        if test_database(i,end)<test_database(j,end)
            DECISION_TRUE(i,j)=1;
        end
    end
end

NMP=0;
for i=1:M
    for j=1:M
        if ~isempty(IN_MARK)&&~isempty(DE_MARK)&&IN_TRUE(i,j)==1&&DE_TRUE(i,j)==1&&DECISION_TRUE(i,j)==1
            NMP=NMP+1;
        end
        if ~isempty(IN_MARK)&&isempty(DE_MARK)&&IN_TRUE(i,j)==1&&DECISION_TRUE(i,j)==1
            NMP=NMP+1;
        end
        if isempty(IN_MARK)&&~isempty(DE_MARK)&&DE_TRUE(i,j)==1&&DECISION_TRUE(i,j)==1
            NMP=NMP+1;
        end
    end
end

NMI=NMP/(M^2-M);
Prediction(Num_test,:)=[];
LABLE(Num_test,:)=[];
disp('不能预测的样本个数为：');
disp(Num);
if type==REGRESSION
    RMSE=sqrt((sum((Prediction-LABLE).^2))/(M-Num));
    disp('RMSE is');
    disp(RMSE);
    MAE=sum(sum(abs(Prediction-LABLE)))/(M-Num);
    disp('MAE is');
    disp(MAE);
    disp('NMI is');
    disp(NMI);
elseif type==CLASSIFIER
    num=0;
    for i=1:M-Num
        if Prediction(i)==LABLE(i)
            num=num+1;
        end
    end
    Prediction_Accuracy=num/(M-Num);
    disp('Prediction_Accuracy is');
    disp(Prediction_Accuracy);
    MAE=sum(sum(abs(Prediction-LABLE)))/(M-Num);
    disp('MAE is');
    disp(MAE);
    disp('NMI is');
    disp(NMI);
end
end                                
