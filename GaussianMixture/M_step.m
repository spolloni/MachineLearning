%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute parameters mu, sigma, pi as a function
% of responsabilitites (gammas) and data x.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mu,sigma,pi] = M_step(gamma,x)

    % dimensions;
    n = size(gamma,1);
    k = size(gamma,2);

    % compute N_k;
    N_k = sum(gamma,1);
    
    % transform data;
    X = repmat(permute(x,[1,3,2]),1,k,1);
    
    % compute new mu;
    mu_p = sum(repmat(gamma,1,1,2).*X)./repmat(N_k,1,1,2);
    mu   = permute(mu_p,[3,2,1]);
    
    % compute new sigma;
    diff = X - repmat(mu_p,n,1,1);
    for i=1:k
        sub_diff = permute(diff(:,i,:),[3,1,2]);
        sigma(:,:,i) = (repmat(gamma(:,i),1,2)'.*sub_diff)*sub_diff';
        sigma(:,:,i) = sigma(:,:,i)/N_k(i);    
    end
    
    % compute new pi;
    pi = (N_k/n)';

end
