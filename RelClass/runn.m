function runn(trainfile, testfile, tau, constraint_type, pred_type, use_LDG, seed = 0)
%{
___________INPUT_____________
    trainfile, testfile - (path to) train and test files
    tau - reliability threshold
    eg.: [1e-30 1e-10 1e-5 .001 .1 .25 .9]
    constraint_type - [ 'boxco' | 'Naive' | 'Cheby' ]
    pred_type - [ 'Corr' | 'Cond' ] 
        'Corr': correlated multidimensional Gaussian prediction
        'Cond': uncorrelated class-conditional GMM prediction
    use_LDG - [ 0 | 1 ] whether to use LDG dimensionality reduction
    
    ?TODO?:
    min_d allows to pass a lower bound to the early classification.
    This lower bound can be derived using a previous run with lower
    value for tau. For repeated runs with different values for tau
    the early_t values could be cached.
%}

% add paths
warning('off');
restoredefaultpath;
addpath('Reliable_Early_Classification');
addpath(genpath('Utilities'));
pkg load statistics;
rand("state", seed);

% load data
test = load(testfile);
data.test.labels = test(:,1);
data.test.data = test(:,2:end);
train = load(trainfile);
data.train.labels = train(:,1);
data.train.data = train(:,2:end);
min_d = [];

try
    [early_l, ~, early_t, ~, ~, ~, training_time, testing_time, ~, ~, ~] = ...
        multi_class_incomplete_classification(data, tau, constraint_type, pred_type, use_LDG, min_d);
catch err
    fprintf(2, '%s %s\n',err.identifier, err.message);
    fprintf('Result: SUCCESS, 0, [1, 1], 0');
    exit;
end

min_d = early_t;

% paramILS report
fprintf(...
    'Result: SUCCESS, %g, [%g, %g], 0\n',...
    training_time+testing_time,...                      % runtime
    mean(early_t)/size(data.test.data,2),...            % earliness
    sum(early_l ~= data.test.labels)/length(early_l)... % fraction misclassified
);
end

