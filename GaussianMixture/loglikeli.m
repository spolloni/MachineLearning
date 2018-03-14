%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute loglikelihood given parameters and data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ll = loglikeli(mu,sigma,pi,x)

    % dimensions;
    n = size(x,1);
    k = size(pi,1);
    
    % normal densities;
    for i=1:k
        norms(:,i) = mvnpdf(x,mu(:,i)',sigma(:,:,i));
    end
    
    % loglikelihood
    ll = sum(log(norms*pi),1);
    
end
