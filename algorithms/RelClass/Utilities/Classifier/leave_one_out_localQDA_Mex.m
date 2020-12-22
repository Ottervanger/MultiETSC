function [test_label, expscore] = leave_one_out_localQDA_Mex(prior, train, trainL, kg, D )
%This function implements localQDA, where the covariance matrices for each
%Gaussian are diaganol.  

%Inputs
% prior: matrix of priors for each class.  Set to [] if you want equal
%   priors among all classes
% train: [num_train x dim] matrix of training feature vectors.
% trainL: [num_train x 1] vector of training labels.
% kg: number of near neighbors to use to fit the local Gaussians
% D: Matrix of distances between training data (input [] for D to
%   and the function will compute distances internally)

%Outputs
% test_label: test labels determined by the local QDA algorithm
% expscore: [num_train x num_classes] matrix of QDA scores for each, class
%   smaller is better

%Last edited by Nathan Parrish
%11/9/2011

%Sort the training data according to labels
classes = unique(trainL);
Nclasses = numel(classes);

if isempty(prior)  %If prior is empty, yse the uniform prior
    prior = ones(Nclasses,1)/Nclasses;
end;

%The following mex function computes the local means and variances
[ means, vars] = leave_one_out_QDAGauss( train, trainL, kg, D );

%The following mex function computes the -log likelihood for each class
[expscore] = LocalGaussobj(train', means, vars, prior, Nclasses);
[~,min_class] = min(expscore,[],2);
test_label = classes(min_class);


