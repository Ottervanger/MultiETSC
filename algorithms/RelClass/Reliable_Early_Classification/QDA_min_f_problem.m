function [ minimum ] = QDA_min_f_problem( sigma, x_k, m_u, R_inv , mu_k, mu_u, delta )

%This function computes the maximum QDA score for the Chebyshev constraint
%set and the Naive Bayes (non box) constraint set.  Solves the problem:

%minimize (x-mu)'Sig^(-1)(x-mu)
%s.t. (x_u - m_u)'R_inv(x_u - m_u) <= delta

%using a minorization approach.  This code requires access to Mosek.  If
%Mosek is not available, it can also be solved using CVX, which is slower.
%This code is provided but is commented out.  Alternatively, the problem
%can also be solved using the dual SDP problem, which is convex.  This code
%is provided as well, at the end of the code, but is very slow and is
%commented out.

%___________INPUT_____________
    %prior - prior for this class
    %sigma - diagonal matrix of Sig (QDA covariance matrix)
    %x_k - the known part of the signal x
    %m_u - the mean of the unknown part of the signal
    %R_inv - the inverse covariance matrix of the unknown part of the
        %signal
    %mu_k - the QDA mean of the known part of the signal
    %mu_u - the QDA mean of the unknown part of the signal
    %delta - threshold parameter

%___________OUTPUT_____________
    %maximum - the maximum of the above problem
    
%Last edited by Nathan Parrish 3/5/2012

%Break up the QDA covariance matrix into the part pertaining to the known,
%unknown and cross portions.
S_uu = diag(1./sigma(length(x_k)+1:end));
S_kk = diag(1./sigma(1:length(x_k)));
S_uk = zeros(length(mu_u),length(mu_k));
S_ku = S_uk';

%Constants
b = mu_u' - S_uu^(-1)*S_uk*(x_k - mu_k)';
c = (x_k-mu_k)*S_kk*(x_k-mu_k)'+2*(x_k-mu_k)*S_ku*mu_u'-(x_k-mu_k)*S_ku*S_uu^(-1)*S_uk*(x_k-mu_k)';

%Uncomment the following to use the dual SDP approach
cvx_begin quiet
    cvx_solver sdpt3
    variables gam(1) lam(1)
    maximize (gam);
    subject to
        lam >= 0;
        [S_uu+lam*R_inv -S_uu*b-lam*R_inv*m_u'; (-S_uu*b-lam*R_inv*m_u')' b'*S_uu*b+c+lam*(m_u*R_inv*m_u' - delta)-gam ] == semidefinite( length(m_u) + 1);
cvx_end

minimum = gam;

end

