function [test_label, expscore] = localQDA_Mex(prior, test, train, trainL, kg, D )
%This function implements localQDA, where the covariance matrices for each
%Gaussian are diaganol.  

%Inputs
% prior: matrix of priors for each class.  Set to [] if you want equal
%   priors among all classes
% test: [num_test x dim] matrix of test feature vectors.
% train: [num_train x dim] matrix of training feature vectors.
% trainL: [num_train x 1] vector of training labels.
% kg: number of near neighbors to use to fit the local Gaussians
% D: Matrix of distances between test and training data (input [] for D to
%   and the function will compute distances internally)

%Outputs
% test_label: test labels determined by the local QDA algorithm
% expscore: [num_test x num_classes] matrix of QDA scores for each, class
%   smaller is better

%Last edited by Nathan Parrish
%11/9/2011

%Sort the training data according to labels
classes = unique(trainL);
Nclasses = numel(classes);
[Ntest,dim] = size(test);

if isempty(prior)  %Make sure that the priors align correctly to the classes
    prior = ones(Nclasses,1)/Nclasses;
end;

expscore = zeros( Ntest, Nclasses ); % give score for every class
test_label = zeros(Ntest,1);

% distance between each test point and each training point
if isempty(D)
    D = pdist2( test, train, 'euclidean' );
end;

%The following mex function computes the local means and variances
[ means, vars] = localGauss_QDA_mex( test, train, trainL, kg, D, eye(dim) );

%The following mex function computes the -log likelihood for each class
[expscore] = LocalGaussobj(test', means, vars, prior, Nclasses);
[~,min_class] = min(expscore,[],2);
test_label = classes(min_class);


