% unfinisheed code to perform LDA
% Need Statistics and Machine Learning toolbox

load ftr.mat

% Data Selection (80/20)
k = 800;
ftr1 = feature_matrix([1:k, 801:(800+k)], :);
all = randperm(size(ftr1,1));
train = all(1:round(0.8*size(ftr1,1)));
test = all(round(0.8*size(ftr1,1))+1:end);

X_train = ftr1(train,1:216);
Y_train = ftr1(train,217);
X_test = ftr1(test,1:216);
Y_test = ftr1(test,217);

Mdl = fitcdiscr(X_train,Y_train)
