%% This file extracts feature matrix and saves it as ftr.mat
% The input set contains 800 positive class and 800 negative class.


clear all

itr = 1;

% For positive class
files = dir('../columbia-prcg-datasets/google_images/*.jpg');
% files = dir('../columbia-prcg-datasets/goog_test/*.jpg')

dir_addr = '../columbia-prcg-datasets/google_images/'
for file = files'
    im = imread(strcat(dir_addr, file.name));
    ftr(itr,:) = [cgorphoto( im ), 1];
    itr = itr+1
    % Do some stuff
end

% For negative class
files = dir('../columbia-prcg-datasets/prcg_images/*.jpg')
dir_addr = '../columbia-prcg-datasets/prcg_images/'
for file = files'
    im = imread(strcat(dir_addr, file.name));
    ftr(itr,:) = [cgorphoto( im ), 0];
    itr = itr+1
    % Do some stuff
end

save(ftr.mat, ftr); 