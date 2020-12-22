function [ means, vars ] = leave_one_out_QDAGauss( X, y, num_neighbors, D )
%This function fits a Gaussian distribution for each class represented in
%y_train to each vector in X_test using the local neighbors from X_train.

%Inputs
% X - [n x dim] matrix of training data
% y - [n x 1] vector of training labels
% num_neighbors - number of neighbors to fit the local Gaussians
% D (Optional) - [n x n] pre-computed distance matrix between training data

%Outputs
% means - matrix of means
% vars - matrix of variances

%Written by Nathan Parrish
%Last edited 1/26/2012

classes = unique(y);

if isempty(D)
    D = pdist2( X, X, 'euclidean');
end;

diag_D = inf* ones(length(y),1);
D = D + diag(diag_D);
clear diag_D;
D = D';
[~,IX] = sort(D,1);
IX(end,:) = [];

[means, vars] = MeansVarsforQDA(X', y, IX, num_neighbors, classes);
end