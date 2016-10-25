clear all;
load ftr.mat

% SVM classification
% This list is being made to find the average random accuracy (for 80/20 selection).  


avg_over = 100;
acc = zeros(length(avg_over),1);
for index = 1:avg_over
    k = 800
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
    acc(index) = mean(pred == Y_test);
    
end

%plot(k_list, acc);
display('The mean accuracy is: ');
mean(acc);