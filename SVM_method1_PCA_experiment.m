clear all;
load ftr.mat

% SVM classification with PCA
% This list is being made to find the average random accuracy (for 80/20 selection).  

avg_over = 15;
acc = zeros(length(avg_over),1);
acc_pca = zeros(length(avg_over),1);
feature_matrix = ftr;

for index = 1:avg_over
    k = 800;
    ftr1 = feature_matrix([1:k, 801:(800+k)], :);
    all = randperm(size(ftr1,1));
    train = all(1:round(0.8*size(ftr1,1)));
    test = all(round(0.8*size(ftr1,1))+1:end);
    
    
    X_train = ftr1(train,1:216);
    Y_train = ftr1(train,217);
    X_test = ftr1(test,1:216);
    Y_test = ftr1(test,217);
    
    %% Perform PCA of training data and test data
    C = 1/(size(X_train,1))*X_train'*X_train;
    keep_dims = 125;
    [U, D] = eigs(C,keep_dims, 'lm');
    
    X_train_pca = X_train*U;
    X_test_pca = X_test*U;
    
    %% Training
    
    SVMStruct = svmtrain(X_train,Y_train, 'Kernel_Function','polynomial');
    SVMStruct_pca = svmtrain(X_train_pca,Y_train, 'Kernel_Function','polynomial');
    
    %% Testing
    
    pred = svmclassify(SVMStruct,X_test);
    pred_pca = svmclassify(SVMStruct_pca,X_test_pca);
    
    %% Evaluation
    acc(index) = mean(pred == Y_test);
    acc_pca(index) = mean(pred_pca == Y_test);
    
%     % Accuracy for positive class
%     p_subset = find(Y_test == 1);
%     acc_pos(index) = mean(pred(p_subset)==1);
%     
%     % Accuracy for negative class
%     p_subset = find(Y_test == 0);
%     acc_neg(index) = mean(pred(p_subset)==0);
    
end

%plot(k_list, acc);
display('The mean accuracy is: ');
 mean(acc)
% mean(acc_pos)
% mean(acc_neg)

plot(acc, 'r')
hold on
plot(acc_pca, 'g');
mean(acc_pca)