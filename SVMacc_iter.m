clear all;
load ftr.mat;

feature_matrix = ftr;

%%
k_list = 50:5:800;
acc = zeros(length(k_list),1);
index = 1;
for k = k_list
    
    ftr1 = feature_matrix([1:k, 801:(800+k)], :);
    all = randperm(size(ftr1,1));
    train = all(1:round(0.8*size(ftr1,1)));
    test = all(round(0.8*size(ftr1,1))+1:end);
    
    
    X_train = ftr1(train,1:216);
    Y_train = ftr1(train,217);
    X_test = ftr1(test,1:216);
    Y_test = ftr1(test,217);
    
    
    %% Training
    
    SVMStruct = svmtrain(X_train,Y_train, 'Kernel_Function','polynomial');
    
    %% Testing
    
    pred = svmclassify(SVMStruct,X_test);
    
    
    %% Evaluation
    acc(index) = sum(pred == Y_test)/length(Y_test)*100
    index = index+1;
    
end

plot(k_list, smooth(smooth(smooth(smooth(smooth(acc))))));
