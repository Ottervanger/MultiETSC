function [ B_LDG, gam_selected ] = LDG_for_Early( neighbs_for_valid, x_t, y_t, num_final_dims, gam, B_Init, neighbs_for_means )
%This function performs Local Discriminative Gaussian (LDG) dimensionality
%reduction.

%Inputs:
    % neighbs_for_valid: number of neighbors to use in the knn
    % cross-validation criteria
    % x_t: [n_t x dim] matrix, each row is a training feature vector
    % y_t: [n_t x 1] vector of training labels
    % num_final_dims: number of final dimensions  
    % gam: vector of values over which to cross-validate gamma
    % neighbs_for_means: optional, number of neighbors to use for LDG.  If
    % this is not entered, then it will be chosen by cross validation.

%Outputs:
    %B_LDG: [dim x num_final_dims] output LDG matrix
    %gam_selected: gamma value chosen by cross validation
    
%Written by Nathan Parrish
%Last edited 11/09/2011

[num_tr, dim] = size(x_t);
classes = unique(y_t);
num_classes = length(classes);
%Make the classes numberd 1 through num_classes.  This is necessary for the
%DiscMeansVars mex code, which should probably be fixed so that this isn't
%necessary.
tr_l = zeros(num_tr,1);
for counter = 1:num_classes
    tr_l(y_t == classes(counter)) = counter;
end;
y_t = tr_l;
clear tr_l;

classes = unique(y_t);
num_classes = length(classes);

priors = zeros(num_classes,1);
for counter = 1:num_classes
    priors(counter) = sum(y_t == classes(counter))/length(y_t);
end;

if isempty(B_Init)
    B_Init = eye(dim);
end;

if ~exist('neighbs_for_means', 'var')
    
    cv_neighbor_size = [4 8 16 32 64 128];
    cv_acc_QDA = zeros(1,length(cv_neighbor_size));
    D = pdist2( x_t, x_t, 'euclidean');

    %Choose the number of neighbors for LDG
    for count_cv = 1:length(cv_neighbor_size)
        [cv_label, ~] = leave_one_out_localQDA_Mex(priors, x_t, y_t, cv_neighbor_size(count_cv), D );
        cv_acc_QDA(count_cv) = sum(y_t == cv_label)/length(y_t)*100;
    end;

    [~, neighbs_for_means] = find(cv_acc_QDA == max(cv_acc_QDA));
    neighbs_for_means = cv_neighbor_size(max(neighbs_for_means));
    clear D cv_neighbor_size cv_acc_QDA cv_ind test testL train trainL test_label
end;

%Create the local Guassians for each data vector in x_t.  We have found
%that using the identity for the covariances works well, which is why we
%set local_vars and local_vars disc the way we do.
[local_means, ~, local_means_disc, ~] = localGauss_LDG( x_t, y_t, neighbs_for_means, B_Init);
local_vars = ones(num_tr,1);
local_vars_disc = ones(num_tr,1,num_classes-1);

nan_elements = isnan(local_means(:,1));
local_means(nan_elements,:) = x_t(nan_elements,:);


%Create the inner product matrices
delta_now = (x_t - local_means)./repmat(sqrt(local_vars), [1,dim]);
V = delta_now.'*delta_now;
delta_now = delta_now.*repmat(sqrt(priors(y_t)),[1,dim]);
A = delta_now.'*delta_now;

%Indexing matrix for the rest of the computation of A
classes_idx = repmat( classes', [num_tr,1])';
classes_idx(y_t + ((1:num_tr)-1)'*num_classes) = [];
classes_idx = reshape(classes_idx, num_classes-1, [])';
for counter = 1:size(local_means_disc,3)
    delta_now = (x_t - local_means_disc(:,:,counter))./repmat(sqrt(local_vars_disc(:,:,counter)), [1,dim]).*repmat(sqrt(priors(classes_idx(:,counter))), [1,dim]);
    A = A + delta_now.'*delta_now;
end;

%%%%%%%%% Memory Saving  %%%%%%%%%%%%%
clear local_means local_means_disc delta_now

%Iterate over the values of gamma
B_hold = zeros(dim,num_final_dims,length(gam));
cv_acc_B = zeros(length(gam),1);
check_dims = unique(round(linspace(2,num_final_dims,10)));
for counter_gam = 1:length(gam)
    %Find the top eigenvectors of (gamma*A - V)
    Z = -(V - gam(counter_gam)*A);
    opts.disp = 0;
    if num_final_dims < size(x_t,2)
        try
            [B,D] = eigs(Z, [], num_final_dims, 'LA', opts);
        catch
            [B,D] = eig(Z);
        end;
    else
        [B,D] = eig(Z);
    end;
    D = (diag(D));
    ix = imag(D) ~= 0;
    B(:,ix) = 0;
    D(ix) = -inf;
    [~,ix] = sort(D,'descend');
    B= B(:,ix(1:num_final_dims));
    %Compute the cross-validation accuracy for this value of gamma, unless
    if length(gam) > 1
        for counter_check = 1:length(check_dims)
            %[labels, ~] = leave_one_out_KNN(y_t, x_t*B(:,1:min(num_final_dims,10)), neighbs_for_valid );
            [labels, ~] = leave_one_out_localQDA_Mex(priors, x_t*B(:,1:check_dims(counter_check)), y_t, neighbs_for_valid, [] );
            cv_acc_B(counter_gam) = max(cv_acc_B(counter_gam),sum(labels == y_t)/length(y_t)*100);
        end;
    end;
    B_hold(:,:,counter_gam) = B;
    clear B Z D;

end;

%Choose the B associated with the largest cv accuracy
ix = find(cv_acc_B == max(cv_acc_B));
ix = max(ix);  %In the case of ties, go with the largest value of gamma
if isempty(ix);
    ix = length(gam);
end;

gam_selected = gam(ix);
B_LDG = B_hold(:,:,ix);

end

