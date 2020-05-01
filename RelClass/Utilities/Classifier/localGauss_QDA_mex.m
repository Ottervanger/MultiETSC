function [ means, vars ] = localGauss_QDA_mex( X_test, X_train, y_train, num_neighbors, D, B )
%This function fits a Gaussian distribution for each class represented in
%y_train to each vector in X_test using the local neighbors from X_train.

%Inputs
% X_test - [n_t x dim] matrix of test data
% X_train - [n_s x dim] matrix of training data
% y_train - [n_s x 1] vector of training labels
% num_neighbors - number of neighbors to fit the local Gaussians
% D (Optional) - [n_t x n_s] pre-computed distance matrix between target 
%   and source domain feature vectors used to determine nearest neighbors.
%   Set to [] if you want to use the Euclidean distance.
% B - [dim x k] linear dimensionality reduction matrix.  Set to $B$ if you
%   want no dimensionality reduction (ie. Euclidian Distance).

%Outputs
% means - matrix of means, the first n_t columns are the class1 means for
    % each test signal, the next n_t columsn are the class2 means for each test
    % signal, etc.
% vars - matrix of variances, the first n_t columns are the class1 variances for
    % each test signal, the next n_t columsn are the class2 variances for each test
    % signal, etc.

%Written by Nathan Parrish
%Last edited 1/26/2012

[num_test, dim] = size(X_test);
classes = unique(y_train);

if isempty(B)
    B = eye(dim);
end;

if isempty(D)
    D = pdist2( X_test*B, X_train*B, 'euclidean' );  %Distances between training and test data
end;

D = D';
[~,IX] = sort(D);


[means, vars] = MeansVarsforQDA(X_train', y_train, IX, num_neighbors, classes);


end