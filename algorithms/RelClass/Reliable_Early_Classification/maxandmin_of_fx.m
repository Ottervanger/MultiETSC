function [f_min,f_max] = maxandmin_of_fx(prior, test, testVar, train, trainL, kg, BESTWORSTCONF, num_unknown, constraint_type, pred_type)
%This function computes the maximum and minimum of f(x) = q_1(x) - q_2(x)
%for a two-class problem.

%___________INPUT_____________
    %prior - vector of priors for each class
    %test - [num_test x dim] matrix of test signal means
    %testVar - [num_unknown x num_unknown] matrix of test signal variances
        %This is a single matrix that serves as the variance for every test
        %signal
    %train - matrix of training data
    %trainL - training labels
    %kg - number of local nearest neighbors to choose for the localQDA
    %BESTWORSTCONF - requirement for the percentage of early labels to
        %match final labels
    %num_unknown - number of time steps remaining unseen in the test data
    %constraint_type - 'Cheby','Naive','BoxCo' for the Chebyshev constraint
        %set, Naive Bayes constraint set, or Naive Bayes box constraint set
    %pred_type - 'Corr', 'Cond' for correlated Gaussian prediction or Class
        %Conditional uncorrelated GMM prediction, respectively

%___________OUTPUT_____________
    %f_max - maximum of q_{-1}(x) - q_1(x)
    %f_min - minimum of q_{-1}(x) - q_1(x)
    
%Last edited by Nathan Parrish 2/21/2012

[trainL,idx] = sort(trainL);
train = train(idx,:);

classes = unique(trainL);
Nclasses = numel(classes);
if Nclasses > 2
    error('This function only works for 2 class problems');
end;
[Ntest,dim] = size(test);

if ~exist('BESTWORSTCONF','var')
    BESTWORSTCONF = 0.95; % with 95% confidence
end

%Thresholds for the Chebyshev inequality constraint set and the Naive Bayes
%Gaussian constraint set
thresh_Cheby = num_unknown / (1-BESTWORSTCONF); 
thresh_Gauss = chi2inv(BESTWORSTCONF, num_unknown);

if isempty(testVar)
    testVar = zeros(size(test));
end

%Find the QDA means and diaganol covariance matrices.
[ m, S ] = localGauss_QDA_mex( test, train, trainL, kg, [], [] );
mneg1 = m(:,1:Ntest)';  %Class -1 QDA means (by row)
m1 = m(:,Ntest+1:end)'; %Class 1 QDA means (by row)
Sneg1 = S(:,1:Ntest)'; %Class -1 QDA vars (by row)
S1 = S(:,Ntest+1:end)'; %Class 1 QDA vars(by row);
pneg1 = prior(1);
p1 = prior(2);

if strcmpi(pred_type,'corr')
    %Precompute the inverse of the test data covariance matrix
    R_inv = pinv(testVar);
    var_for_box = testVar;
end;

f_min = zeros(Ntest,1);
f_max = zeros(Ntest,1);

for n=1:Ntest,
    
    %% Precomputations
    C2 = sum(log(Sneg1(n,:))) - 2*log(pneg1) - sum(log(S1(n,:))) + 2*log(p1);
    Sell = -S1(n,:);
    Sig_d_inv = (1./Sneg1(n,:) + 1./Sell);  %Sig_d = (Sig_{-1}^{-1} - Sig_1^{-1})^{-1}
    Sig_d_inv(Sig_d_inv == 0) = 1e-10; %This is currently a hack that makes for numerical stability.
    Sig_d = 1./Sig_d_inv;
    diff_vec = ( (1./Sneg1(n,:)).*mneg1(n,:) + (1./Sell).*m1(n,:) );
    d = Sig_d.*diff_vec; %m_d = Sig_d(Sig_{-1}^{-1} mu_{-1} - Sig_{1}^{-1} mu_{1});
    C1 = - diff_vec*(d)' + ( mneg1(n,:)*(1./Sneg1(n,:).*mneg1(n,:))' + m1(n,:)*(1./Sell.*m1(n,:))');
    
    %% Find the maximum and minimum of f(x) for the current test data
    
    if strcmpi(pred_type,'cond')
        if ~strcmpi(constraint_type, 'cheby')  %No dimensionality reduction
            R_inv = diag(1./(testVar(n,dim-num_unknown+1:end)));
            var_for_box = diag(testVar(n,dim-num_unknown+1:end));
        else
            R_inv = pinv(testVar(dim-num_unknown+1:end,dim-num_unknown+1:end,n)); 
        end;
    end;
    
    if strcmpi(constraint_type,'cheby') %Chebyshev constraint set
        %The following two lines implement the slower SDP solution
%         f_max(n) = QDA_max_f_problem( Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), R_inv , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), thresh_Cheby );
%         f_max(n) = f_max(n) + C1 + C2;
        %The following line implements the faster DC solution
        f_max(n) = -QDA_min_f_DC( -Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), R_inv , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), thresh_Cheby, -C1 - C2 );
        if f_max(n) > 0
            %The following two lines implement the slower SDP solution
%             f_min(n) = QDA_min_f_problem( Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), R_inv , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), thresh_Cheby );
%             f_min(n) = f_min(n) + C1 + C2;
            %The following line implements the faster DC solution
            f_min(n) = QDA_min_f_DC( Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), R_inv , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), thresh_Cheby, C1 + C2 );
        end;
    elseif strcmpi(constraint_type,'naive') %Naive Bayes Gaussian Constraint Set
        %The following two lines implement the slower SDP solution
%         f_max(n) = QDA_max_f_problem( Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), R_inv , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), thresh_Gauss );
%         f_max(n) = f_max(n) + C1 + C2;
        %The following line implements the faster DC solution
        f_max(n) = -QDA_min_f_DC( -Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), R_inv , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), thresh_Gauss, -C1 -C2 );
        if f_max(n) > 0
            %The following two lines implement the slower SDP solution
%             f_min(n) = QDA_min_f_problem( Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), R_inv , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), thresh_Gauss );
%             f_min(n) = f_min(n) + C1 + C2;
            %The following line implements the faster DC solution
            f_min(n) = QDA_min_f_DC( Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), R_inv , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), thresh_Gauss, C1 + C2 );
        end;
    elseif strcmpi(constraint_type,'boxco') %Naive Bayes Gaussian Box Constraint Set
    %Gaussian box constraint     
        f_max(n) = max_box_constraints_f_problem( Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), var_for_box , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), BESTWORSTCONF );
        f_max(n) = f_max(n) + C1 + C2;
        if f_max(n) > 0
            f_min(n) = min_box_constraints_f_problem( Sig_d , test(n,1:dim-num_unknown), test(n,dim-num_unknown+1:end), var_for_box , d(1:dim-num_unknown), d( dim-num_unknown+1:end ), BESTWORSTCONF );
            f_min(n) = f_min(n) + C1 + C2;
        end;
    end;
    
    

end



