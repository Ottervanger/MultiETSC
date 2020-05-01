function [ minimum ] = QDA_min_f_DC( sigma, x_k, m_u, R_inv , mu_k, mu_u, delta, c_extern )

%This function computes the minimum of a quadratic function over a
%quadratic constraint set using the difference of convex functions approach
%found in the paper "CONVEX ANALYSIS APPROACH TO D. C. PROGRAMMING:
%THEORY, ALGORITHMS AND APPLICATIONS"

%The problem is: 
%minimize (x-mu)'sigma^(-1)(x-mu)
%s.t. (x_u - m_u)'R_inv(x_u - m_u) <= delta

%___________INPUT_____________
    %sigma - DIAGONAL matrix (QDA covariance matrix)
    %x_k - the known part of the signal x
    %m_u - the mean of the unknown part of the signal
    %R_inv - the inverse covariance matrix of the unknown part of the signal
    %mu_k - the QDA mean of the known part of the signal
    %mu_u - the QDA mean of the unknown part of the signal
    %delta - threshold parameter
    %c_extern - this is an external constant that is added to the final
    %solution

%___________OUTPUT_____________
    %minimum - the maximum of the above problem
    
%Last edited by Nathan Parrish 4/14/2012

dim = length(m_u);

%Break up the QDA covariance matrix into the part pertaining to the known,
%unknown and cross portions.
S_uu = diag(1./sigma(length(x_k)+1:end));
S_kk = diag(1./sigma(1:length(x_k)));
S_uk = zeros(length(mu_u),length(mu_k));
S_ku = S_uk';

%Constants
d = mu_u' - pinv(S_uu)*S_uk*(x_k - mu_k)';
c = (x_k-mu_k)*S_kk*(x_k-mu_k)'+2*(x_k-mu_k)*S_ku*mu_u'-(x_k-mu_k)*S_ku*pinv(S_uu)*S_uk*(x_k-mu_k)'+c_extern;

% At this point, the problem that we must solve is:
% minimize (x_u - d)'S_uu(x_u -d) + c
% s.t.     (x_u - m_u)'R_inv(x_u - m_u) <= delta

%We now want to transform this into a standard trust region problem of the
%form:
%minimize 1/2z'Az + b'z + c_net
%s.t.     ||z|| <= r

R = pinv(R_inv);
% We do this by defininig the following:
%z = R^(-1/2)*(x_u' - m_u');
A = 2*sqrtm(R)*S_uu*sqrtm(R);
b = 2*(sqrtm(R)*S_uu*(m_u'-d));
r = sqrt(delta);
c_net = c + d'*S_uu*d + m_u*S_uu*m_u' -2*m_u*S_uu*d;

%Set rho equal to the maximal eigenvalue of A
% try
%     rho = eigs(A,[],1,'la');
% catch
%     rho = eigs(A,[],1,'lr');
% end;
rho = eig(A);
rho = max(rho);
rho = max(1,rho);

%Now, we iterate to find the solution
rat = 1;  %Iterate until the value of rat is small enough
z_now = ones(dim,1)*sqrt(delta)/sqrt(dim);  %Intialize
%hold_z = z_now';
hold_sol = 1/2*z_now'*A*z_now + b'*z_now + c_net;

while rat > 1e-2
    
    z_now = (z_now - 1/rho*(A*z_now + b));    
    
    if norm(z_now) > r
        z_now = z_now * r/norm(z_now);
    end;
    
    %hold_z = [hold_z;z_now'];
    hold_sol = [hold_sol; 1/2*z_now'*A*z_now + b'*z_now + c_net];
    
    if length(hold_sol) > 2
        rat = ((hold_sol(end -1) - hold_sol(end)))/abs(hold_sol(end-1));
    end;
    
    %If the current solution ever descends below zero, then we need go no
    %further
    if (hold_sol(end)) < 0
        break
    end;
end

minimum = hold_sol(end);



