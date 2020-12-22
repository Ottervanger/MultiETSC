function [ minimum ] = min_box_constraints_f_problem( sigma, x_k, m_u, R , mu_k, mu_u, delta )
%This function computes the minimum f(x) score for the Naive Bayes box
%constraint.  Solves the problem:

%minimize (x-mu)'sigma^(-1)(x-mu)
%s.t. |x_u(l) - m_u(l)| <= spacing(l)  for all l = 1...n

%where n is the length of x_u and spacing(l) is a parameter that assigns 
%delta^(1/n) probability to the covariate x_u(l).  Since Sig is diaganol,
%the problem can be solved for each x_u(l) individually.  If Sig(u,u) < 0,
%place x_u(l) as far away from mu_u(l) as possible.  If Sig(u,u) >= 0,
%place x_u(l) as close to mu_u(l) as possible.

%___________INPUT_____________
    %sigma - diagonal matrix of Sig (QDA covariance matrix)
    %x_k - the known part of the signal x
    %m_u - the mean of the unknown part of the signal
    %R - the covariance matrix of the unknown part of the
        %signal
    %mu_k - the QDA mean of the known part of the signal
    %mu_u - the QDA mean of the unknown part of the signal
    %delta - threshold parameter

%___________OUTPUT_____________
    %minimum - the minimum of the above problem
    
%Last edited by Nathan Parrish 9/6/2011

%Break up sigma according to the known and unknown portions of the signal
S_uu = diag(1./sigma(length(x_k)+1:end));
S_kk = diag(1./sigma(1:length(x_k)));
S_uk = zeros(length(mu_u),length(mu_k));
S_ku = S_uk';

%Constants
b = mu_u' - pinv(S_uu)*S_uk*(x_k - mu_k)';
c = (x_k-mu_k)*S_kk*(x_k-mu_k)'+2*(x_k-mu_k)*S_ku*mu_u'-(x_k-mu_k)*S_ku*pinv(S_uu)*S_uk*(x_k-mu_k)';
    
r = diag(R)';  %This uses the Naive Bayes assumption that R is diaganol
%Determine the edge of the box for covariate using the Gaussian assumption
delta_per = nthroot(delta, length(m_u));
spacing = erfcinv(1-delta_per);
spacing = spacing/sqrt(1/2)*sqrt(r);

%Compute the upper and lower bounds for each element of x
upper_x = m_u + spacing;
lower_x = m_u - spacing;

%% Placeholder for x_u
x_u = zeros(1,length(m_u));

%% Determine which elements of S_uu are negative, and which are positive
s_diag = diag(S_uu);
pos_elements = s_diag > 0;
neg_elements = ~pos_elements;

%% First, set x_u for the negative elements
%Set element n of x_u at m_u(n) + spacing if b < m_u and m_u(n) - spacing
%otherwise
add_elements = b' <= (upper_x+lower_x)/2;

x_u(add_elements & neg_elements') = upper_x(add_elements & neg_elements');
x_u(~add_elements & neg_elements') = lower_x(~add_elements & neg_elements');
    
%% Now, set x_u for the positive elements
in_elements = and(lower_x < b', upper_x > b');
x_u(in_elements & pos_elements') = (b(in_elements & pos_elements'))';

%Set element n of x_u at m_u(n) + spacing if b(n) > m_u(n) + spacing(n) and
%m_u(n) - spacing(n) if b(n) < m_u(n) - spacing(n)
add_elements = b' > upper_x;
sub_elements = b' < lower_x; 

x_u(add_elements & pos_elements') = upper_x(add_elements & pos_elements');
x_u(sub_elements & pos_elements') = lower_x(sub_elements & pos_elements');

minimum = (x_u' - b)'*S_uu*(x_u' - b) + c;


end

