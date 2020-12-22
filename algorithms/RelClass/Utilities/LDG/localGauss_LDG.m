function [ local_means, local_vars, local_means_disc, local_vars_disc ] = localGauss_LDG( x, y, num_neighbors, B_init )
%This function fits a Gaussian distribution to each vector in x 
%using the local neighbors.  The Gaussians given by local_means
%and local_vars are fit using the same class data.  The Gaussians given by
%local_means_disc and local_vars_disc are fit using the different class
%data.

%Inputs
% x - [n_t x dim] matrix of training data
% y_target - [n_t x 1] vector of labels
% num_neighbors - number of neighbors to fit the local Gaussians

%Outputs
% local_means - [n_t x dim] matrix of same class means
% local_vars - [n_t x dim] matrix of same class covariances
% local_means_disc - [n_t x dim x 1] matrix of different class means means
% local_vars_disc - [n_t x dim x num_classes-1] matrix of different class covariances

%Written by Nathan Parrish
%Last edited 11/09/2011

%Sort the source data according to labels
[n_t, dim] = size(x);
classes = unique(y);
num_classes = length(classes);

if ~exist('B_init', 'var')
    B_init = eye(dim);
end;

D = pdist2( x*B_init, x*B_init, 'euclidean' );

diag_D = inf*ones(n_t,1);
diag_D = diag(diag_D);
D = D + diag_D;
clear diag_D;

[~,IX] = sort(D,1);
IX = IX(1:end-1,:);  %leave out the last index

[local_means, local_vars, means_disc, vars_disc] = DiscMeansVars(x',y,y,IX,num_neighbors, num_classes, classes);

local_means = local_means';
local_vars = local_vars';

local_means_disc = zeros(n_t, dim, num_classes-1);
local_vars_disc = zeros(n_t, dim, num_classes -1);

for counter = 1:num_classes - 1;
    local_means_disc(:,:,counter) = means_disc(:,(counter-1)*n_t+1:(counter*n_t))';
    local_vars_disc(:,:,counter) = vars_disc(:,(counter-1)*n_t+1:(counter*n_t))';
end;

end