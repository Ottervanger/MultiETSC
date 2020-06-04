function [early_l, early_l_sct, early_t, final_l, all_l, ts_l, training_time, testing_time, BESTWORSTCONF, final_dims, LDG_necessary] = multi_class_incomplete_classification( data, BESTWORSTCONF, constraint_type, pred_type, dim_red, min_d )

%This function performs an early classification experiment.  The inputs are
%as follows:

%___________INPUT_____________
    %data - stuct containing test and train fields each containing labels and
        % data fields: the dataset on which you would like to perform the experiment.
    %BESTWORSTCONF - This is the lower bound on the percentage of early labels 
        %that match the final labels - the $\tau$ parameter in the paper
    %constraint_type - 'Cheby','Naive','BoxCo' for the Chebyshev constraint
        %set, Naive Bayes constraint set, or Naive Bayes box constraint set
    %pred_type - 'Corr', 'Cond' for correlated Gaussian prediction or Class
        %Conditional GMM prediction, respectively
    %dim_red - 1 if you would like to perform LDG dimensionality reduction,
        %0 otherwise
    %min_d - optional vector parameter where the n^th element is the
        %minimum possible early classification time for the n^th test
        %signal.  This is useful if some prior knowledge of the minimum d
        %is known, for instance, if the function is being called
        %iteratively for increasing values of $\tau$, then early times from
        %a smaller value of $\tau$ can be used as min_d for a larger value
        %of $\tau$.  This greatly speeds up the run time of this function.

%___________OUTPUT_____________
    %early_l - the early label for each test sample
    %early_t - the early time for each test sample
    %final_l - the final label for each test sample
    %all_l - the label assigned by local QDA at each time, this will be a
        %matrix of size (number of test samples, length of test data)
    %ts_l - the true test labels
    %training_time - time taken to train the classifier
    %testing_time - time taken to perform incomplete classification on the
        %entire test set
    %BESTWORSTCONF - the BESTWORSTCONF used by the early classifier
    %final_dims - number of final dimensions (useful when LDG dimensionality
        %reduction is used)
    %LDG_necessary - marked as 1 if the training process thinks that
        %dimensionality reduction will improve the classifier error rate, 0
        %otherwise (this is not informative if LDG dimensionality reduction is not used)
    
%Edited by Nathan Parrish 7/30/2012
%Edited by Gilles Ottervanger 2020-06-04

if ~exist('BESTWORSTCONF')
    BESTWORSTCONF = 0.95;
end;

if ~exist('min_d')
    min_d = [];
end;

if ~exist('localQDA_Mex')
    %This needs to point to the folder with the Local QDA code
    addpath('C:\Users\nparrish\Documents\Research\Matlab_Code\Early_Classification_For_Web\Utilities\Classifier');
end;
LDG_necessary = 0;

%% Generate Training and Test Datasets
% create random number seed
RANDS = RandStream.create('mrg32k3a','NumStreams',1,'Seed',1);
OLD = RandStream.setGlobalStream( RANDS );

% load test and training data
tr_d0 = data.train.data;
tr_l  = data.train.labels;
ts_d0 = data.test.data;
ts_l  = data.test.labels;
[Ntest,~] = size(ts_d0);
[Ntrain,~] = size(tr_d0);
classes = unique(tr_l);
Nclasses = numel(classes);

%Compute Class priors
prior = zeros(Nclasses,1);
for counter = 1:Nclasses
    prior(counter) = sum(tr_l == classes(counter))/length(tr_l);
    %prior = 1/Nclasses*ones(Nclasses,1);
end;

%Preprocess and normalize the training and test data

tic;
% normalize by a scalar (shift-invariant)
s = std(tr_d0);
where_non_zero = (s ~= 0);
%Delete columns with zero variance
tr_d0 = tr_d0(:,where_non_zero);
ts_d0 = ts_d0(:,where_non_zero);
s = s(where_non_zero);
m = mean(tr_d0);
tr_d0 = (tr_d0 - repmat(m,[Ntrain,1]))./repmat(s, [Ntrain,1]);
ts_d0 = (ts_d0 - repmat(m,[Ntest,1]))./repmat(s, [Ntest,1]);

dim = sum(where_non_zero);

%% Perform Leave-one-out cross-validation to choose the number of neighbors for local QDA
knn_check = [1 2 4 6 8 16 32 64 128 256]; % # of nearest neighbors
knn_check(knn_check > Ntrain/Nclasses) = [];

cv_indcx = crossvalind('Kfold',tr_l,5); %Five fold cv indices for LDG

errors_by_k = zeros(length(knn_check),1);
Dists = pdist2(tr_d0,tr_d0,'euclidean');
for count_cv = 1:5
    tr_d_now = tr_d0(cv_indcx ~= count_cv,:);
    tr_l_now = tr_l(cv_indcx ~= count_cv,:);
    ts_d_now = tr_d0(cv_indcx == count_cv,:);
    ts_l_now = tr_l(cv_indcx == count_cv,:);
    counter_neighbs = 0;
    for neighbs = knn_check
        counter_neighbs = counter_neighbs + 1;
        [labels,~] = localQDA_Mex(prior, ts_d_now, tr_d_now, tr_l_now, neighbs, Dists(cv_indcx == count_cv,cv_indcx ~= count_cv) );
        errors_by_k(counter_neighbs) = errors_by_k(counter_neighbs) + sum(labels ~= ts_l_now);
    end;
end;

[dum,ind_knn] = min(errors_by_k);
full_acc = 100-errors_by_k(ind_knn)/length(tr_l)*100;
knn = knn_check(ind_knn);

%% Perform dimensionality reduction
if dim_red
    if ~exist('LDG_for_Early')
        %This needs to point to the folder with the LDG Code
        addpath('C:\Users\nparrish\Documents\Research\Matlab_Code\Early_Classification_For_Web\Utilities\LDG');
    end;
    
    gam_range = 0.9;
    cv_acc = zeros(length(gam_range),dim);  %(i,j) will hold the cv accuracy for gam_range(i) and j dimensions.
    for count_gam = 1:length(gam_range)
        gam_now = gam_range(count_gam);
        for count_cv = 1:5
            %Make the current training/test split
            tr_d_now = tr_d0(cv_indcx ~= count_cv,:);
            tr_l_now = tr_l(cv_indcx ~= count_cv,:);
            ts_d_now = tr_d0(cv_indcx == count_cv,:);
            ts_l_now = tr_l(cv_indcx == count_cv,:);

            [ B_LDG, ~ ] = LDG_for_Early(knn, tr_d_now, tr_l_now, dim, gam_now, [], knn );

            current_acc = zeros(1,dim);
            for counter_dims = 1:dim
                [labels,~] = localQDA_Mex(prior, ts_d_now*B_LDG(:,1:counter_dims), tr_d_now*B_LDG(:,1:counter_dims), tr_l_now, knn, [] );
                current_acc(counter_dims) = sum(labels == ts_l_now)/length(ts_l_now);
                if counter_dims > 25
                    if max(current_acc(1:counter_dims - 25)) >= max(current_acc(counter_dims - 24:counter_dims));
                        break;
                    end;
                end;
                cv_acc(count_gam,counter_dims) = cv_acc(count_gam,counter_dims) + sum(labels == ts_l_now)/length(tr_l)*100;
            end;
        end;
    end;
    
    top_val = max(cv_acc,[],2);
    
    gam_selected = gam_range(top_val == max(top_val));
    gam_selected = max(gam_selected);
    dim_selected = max(find(cv_acc(gam_selected == gam_range,:) == max(top_val)));
    
    [ B_LDG, ~ ] = LDG_for_Early(knn, tr_d0, tr_l, dim_selected, gam_selected, [], knn );
    
    errors_by_k = zeros(length(knn_check),1);
    Dists = pdist2(tr_d0*B_LDG,tr_d0*B_LDG,'euclidean');
    for count_cv = 1:5
        tr_d_now = tr_d0(cv_indcx ~= count_cv,:)*B_LDG;
        tr_l_now = tr_l(cv_indcx ~= count_cv,:);
        ts_d_now = tr_d0(cv_indcx == count_cv,:)*B_LDG;
        ts_l_now = tr_l(cv_indcx == count_cv,:);
        counter_neighbs = 0;
        for neighbs = knn_check
            counter_neighbs = counter_neighbs + 1;
            [labels,~] = localQDA_Mex(prior, ts_d_now, tr_d_now, tr_l_now, neighbs, Dists(cv_indcx == count_cv,cv_indcx ~= count_cv) );
            errors_by_k(counter_neighbs) = errors_by_k(counter_neighbs) + sum(labels ~= ts_l_now);
        end;
    end;

    [dum,ind_knn] = min(errors_by_k);
    LDG_acc = 100-errors_by_k(ind_knn)/length(tr_l)*100;
    knn = knn_check(ind_knn);
    
    if LDG_acc >= full_acc
        LDG_necessary = 1;
    end;
else
    B_LDG = eye(dim);
end;
final_dims = size(B_LDG,2);
     
clear acc k neighbs CVtrain CVtrainL CVtest CVtestL errors_by_k ind_errors ind_knn Dists  
        
%%  Run the experiment on the test data

early_l = zeros(Ntest,1);
early_l_sct = early_l;
early_t = NaN(Ntest,1);
locknow = zeros(Ntest,1);
%Estimate the global mean and covariance of the training data

if strcmpi(pred_type,'corr')  %Correlated Gaussian prediction
    tr_mean = mean( tr_d0 );
    tr_cov = 0.8*cov(tr_d0)  + .2*diag(var(tr_d0));
else  %Class-conditional GMM prediction
    tr_mean = zeros(Nclasses,dim);  %Matrix for class condtional means, each row corresponds to 1 class
    tr_cov = zeros(dim,dim,Nclasses);  %Matrix for class conditional diaganol covariance matrices, each row corresponds to 1 class
    for count = 1:length(classes)
        tr_mean(count,:) = mean( tr_d0(tr_l == classes(count),:));
        tr_cov(:,:,count) = 0.8*cov( tr_d0(tr_l == classes(count),:)) + 0.2*diag(var( tr_d0(tr_l == classes(count),:)));
    end;
end;

if isempty(min_d)
	min_d = ones(Ntest,1);
end;
all_l = zeros(Ntest,dim);
d=min(min_d);
training_time = toc;
tic;

%This loop steps through the time series data and uses the early
%classification algorithm to determine when to assign an early class label
%for each test signal
while and( d < dim, sum(locknow ~=1 > 0)) % present datapoints 1 timestep at a time
    %disp(num2str(d));
    locknow_idcs = find(and(locknow == 0, min_d <= d));
    
    % present this to the classifier
    [~,sct] = localQDA_Mex(prior, ts_d0(:,1:d), tr_d0(:,1:d), tr_l, knn, [] );
    [~,min_sct_class] = min(sct,[],2);
    all_l(:,d) = classes(min_sct_class);
    
    if strcmpi(pred_type, 'corr')
        thistest = [ts_d0(:,1:d) repmat(tr_mean(d+1:end),[Ntest 1])+(tr_cov(d+1:end,1:d)*pinv(tr_cov(1:d,1:d))*(ts_d0(:,1:d)-repmat(tr_mean(1:d),[Ntest 1]))')' ]*B_LDG; 
        %Joint Gaussian Test Data Estimate
        thistestVar = zeros(dim,dim);
        thistestVar(d+1:end,d+1:end) = (tr_cov(d+1:end,d+1:end) - tr_cov(d+1:end,1:d)*pinv(tr_cov(1:d,1:d))*tr_cov(1:d,d+1:end));
        thistestVar = B_LDG'*thistestVar*B_LDG;
        if strcmpi(constraint_type, 'naive') || strcmpi(constraint_type, 'boxco') %|| strcmpi(constraint_type, 'cheby')
            %Diaganol Covariance Estimate for the Naive Bayes Assumption
            thistestVar = diag(diag( thistestVar)); 
        end;
    else
        if ~strcmpi(constraint_type, 'cheby')
            [ thistest, thistestVar,~ ] = Class_Cond_Prediction( ts_d0(:,1:d), tr_mean, tr_cov, sct, B_LDG  );
            %Submatrix (:,:,i) is the covariance matrix for the ith test
            %signal
        else
            [ thistest, ~, thistestVar ] = Class_Cond_Prediction( ts_d0(:,1:d), tr_mean, tr_cov, sct, B_LDG  );
            %Row i of thistestVar is the diagonal covariance matrix for
            %the ith test signal
        end;
            
    end;        
    
    % The following variable, win_all, is a matrix of size [length(locknow_idcs),
    % Nclasses].  The reason it is only of order locknow_idcs is that we are only
    % concerned with the test signals that have not yet been 'locked'.  
    % The i,j element of this matrix is set to 1 if class j
    % wins comparisons to every other class for test signal locknow_idcs(i).
    %win_all = ones(length(locknow_idcs), Nclasses);
    
    % has_tied(i,j) is one if the j^th class has tied for the i^th signal
    %has_tied = zeros(length(locknow_idcs), Nclasses);
    
    %This part should be commented out if the mean of the test signal is
    %not part of the set A
    %Classify the mean of the test signal
    [~,sct_mean] = localQDA_Mex(prior, thistest(locknow_idcs,:), tr_d0*B_LDG, tr_l, knn, [] );
    [~,min_sct_class_mean] = min(sct_mean,[],2);
    
    %Set win_all to be one for only the class with the mean
    win_all = sparse(1:length(locknow_idcs),min_sct_class_mean,ones(length(locknow_idcs),1));
    win_all = (full(win_all));
    win_all = [win_all zeros(size(win_all,1), Nclasses - size(win_all,2))];
    %Set the classes that do not contain the mean to has_tied = 1
    has_tied = ~win_all;
    
    if 0 %This part is not necessary if the mean is part of the set A
    for class_a = 1:(Nclasses-1)  %Loop over all classes
        for class_b = class_a+1:Nclasses
        
            class_neg_1 = classes(class_a);
            class_pos_1 = classes(class_b);
         
            current_tr_idcs = tr_l == class_neg_1 | tr_l == class_pos_1;
         
            %If one or more of the two current classes is set to
            %needs_compare = 0, then we do not need to compare these two
            %classes for that test signal
            
            check_idcs= find((win_all(:,class_a) & win_all(:,class_b)) == 1);  %This stores the index of the signals that need to be checked where indexing is in terms of the columns of win_all
            check_idcs_test_sigs = locknow_idcs(check_idcs);  %This stores the index of the signals that need to be checked where the indexing is in terms of the test signals
            
            %Compare class a and class b
            f_min = zeros(length(check_idcs_test_sigs),1);
            f_max = zeros(length(check_idcs_test_sigs),1);
            if ~isempty(check_idcs_test_sigs)
                if dim_red == 1
                    if strcmpi(pred_type,'corr')
                        [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar, tr_d0(current_tr_idcs,:)*B_LDG, tr_l(current_tr_idcs), knn, BESTWORSTCONF, size(B_LDG,2), constraint_type, pred_type );
                    else
                        if ~strcmpi(constraint_type,'Cheby')
                            [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(check_idcs_test_sigs,:), tr_d0(current_tr_idcs,:)*B_LDG, tr_l(current_tr_idcs), knn, BESTWORSTCONF, size(B_LDG,2), constraint_type, pred_type );
                        else
                            [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(:,:,check_idcs_test_sigs), tr_d0(current_tr_idcs,:)*B_LDG, tr_l(current_tr_idcs), knn, BESTWORSTCONF, size(B_LDG,2), constraint_type, pred_type );
                        end;
                    end;
                else
                    if strcmpi(pred_type,'corr')
                        [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(d+1:end,d+1:end), tr_d0(current_tr_idcs,:), tr_l(current_tr_idcs), knn, BESTWORSTCONF, dim-d, constraint_type, pred_type );
                    else
                        if ~strcmpi(constraint_type,'Cheby');
                            [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(check_idcs_test_sigs,:), tr_d0(current_tr_idcs,:), tr_l(current_tr_idcs), knn, BESTWORSTCONF, dim-d, constraint_type, pred_type );
                        else
                            [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(:,:,check_idcs_test_sigs), tr_d0(current_tr_idcs,:), tr_l(current_tr_idcs), knn, BESTWORSTCONF, dim-d, constraint_type, pred_type );
                        end;
                    end;
                end;
            end;
            
            %Adjust win_all based on the values of f_min and f_max
            for n = 1:length(check_idcs)
                if f_min(n) > 0 %class_a cannot be the winner
                    win_all(check_idcs(n),class_a) = 0;
                elseif f_max(n) <= 0  %class_b cannot be the winner
                    win_all(check_idcs(n), class_b) = 0;
                else  %Neither class_a nor class_b can be a winner
                    win_all(check_idcs(n),class_a) = 0;
                    win_all(check_idcs(n),class_b) = 0;
                    has_tied(check_idcs(n),class_a) = 1;
                    has_tied(check_idcs(n),class_b) = 1;
                end;
            end;   
        end;
    end;
    end;
    
    %Revisit Ties
    for class_a = 1:(Nclasses-1)  %Loop over all classes
        for class_b = class_a+1:Nclasses

            possible_winners = find(sum(win_all,2) == 1);
            if (sum(sum(has_tied(possible_winners,:)))==0)
                break
            end;

            class_neg_1 = classes(class_a);
            class_pos_1 = classes(class_b);

            current_tr_idcs = tr_l == class_neg_1 | tr_l == class_pos_1;

            %If one or more of the two current classes is set to
            %needs_compare = 0, then we do not need to compare these two
            %classes for that test signal

            check_idcs= find( (win_all(:,class_a) | win_all(:,class_b)) & (has_tied(:,class_a) | has_tied(:,class_b) ) == 1);  %This stores the index of the signals that need to be checked where indexing is in terms of the columns of win_all
            check_idcs_test_sigs = locknow_idcs(check_idcs);  %This stores the index of the signals that need to be checked where the indexing is in terms of the test signals
    
           %Compare class a and class b
            f_min = zeros(length(check_idcs_test_sigs),1);
            f_max = zeros(length(check_idcs_test_sigs),1);
            if ~isempty(check_idcs_test_sigs)
                if dim_red == 1
                    if strcmpi(pred_type,'corr')
                        [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar, tr_d0(current_tr_idcs,:)*B_LDG, tr_l(current_tr_idcs), knn, BESTWORSTCONF, size(B_LDG,2), constraint_type, pred_type );
                    else
                        if ~strcmpi(constraint_type,'Cheby')
                            [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(check_idcs_test_sigs,:), tr_d0(current_tr_idcs,:)*B_LDG, tr_l(current_tr_idcs), knn, BESTWORSTCONF, size(B_LDG,2), constraint_type, pred_type );
                        else
                            [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(:,:,check_idcs_test_sigs), tr_d0(current_tr_idcs,:)*B_LDG, tr_l(current_tr_idcs), knn, BESTWORSTCONF, size(B_LDG,2), constraint_type, pred_type );
                        end;
                    end;
                else
                    if strcmpi(pred_type,'corr')
                        [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(d+1:end,d+1:end), tr_d0(current_tr_idcs,:), tr_l(current_tr_idcs), knn, BESTWORSTCONF, dim-d, constraint_type, pred_type );
                    else
                        if ~strcmpi(constraint_type,'Cheby');
                            [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(check_idcs_test_sigs,:), tr_d0(current_tr_idcs,:), tr_l(current_tr_idcs), knn, BESTWORSTCONF, dim-d, constraint_type, pred_type );
                        else
                            [f_min, f_max] = maxandmin_of_fx(prior([class_a, class_b]), thistest(check_idcs_test_sigs,:), thistestVar(:,:,check_idcs_test_sigs), tr_d0(current_tr_idcs,:), tr_l(current_tr_idcs), knn, BESTWORSTCONF, dim-d, constraint_type, pred_type );
                        end;
                    end;
                end;
            end;

            %Adjust win_all based on the values of f_min and f_max
            for n = 1:length(check_idcs)
                if f_min(n) > 0 %class_a cannot be the winner
                    win_all(check_idcs(n),class_a) = 0;
                elseif f_max(n) <= 0  %class_b cannot be the winner
                    win_all(check_idcs(n), class_b) = 0;
                else  %Neither class_a nor class_b can be a winner
                    win_all(check_idcs(n),class_a) = 0;
                    win_all(check_idcs(n),class_b) = 0;
                    has_tied(check_idcs(n),class_a) = 1;
                    has_tied(check_idcs(n),class_b) = 1;
                end;
            end;
        end;
    end;

    
    % Determine which test signals can be 'locknow', ie assigned an early
    % label, at this iteration.  
    for n = 1:sum(and(locknow == 0, min_d <= d))
        if sum(win_all(n,:)) == 1
            locknow(locknow_idcs(n)) = 1;
            early_t(locknow_idcs(n)) = d;
            early_l(locknow_idcs(n)) = classes(win_all(n,:) == 1);
            early_l_sct(locknow_idcs(n)) = classes(min_sct_class(locknow_idcs(n)));
        end;
    end;   
   
    d = d+1;
end

%% Assign final early labels and compute the final labels
[~,sct] = localQDA_Mex(prior, ts_d0*B_LDG, tr_d0*B_LDG, tr_l, knn, [] );
[mn,tmp] = min(sct,[],2);

% Assign early labels and early times to all test signals that weren't
% classified in the iterations above
mask = isnan(early_t); 
early_t(mask) = dim;
early_l(mask) = classes(tmp(mask));
early_l_sct(mask) = classes(tmp(mask));

%Final labels
final_l = classes(tmp);
testing_time = toc;
