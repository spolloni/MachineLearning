%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute responsabilitites (gammas) as function
% of parameters mu, sigma, pi. and data x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gamma = E_step(mu,sigma,pi,x)

    % dimensions;
    n = size(x,1);
    k = size(pi,1);
    
    % normal densities;
    for i=1:k
        norms(:,i) = mvnpdf(x,mu(:,i)',sigma(:,:,i));
    end
    
    % weight-sum normals;
    sumnorms = repmat(norms*pi,1,k);
    
    % compute responsabilities;
    gamma  =  (repmat(pi',n,1,1).*norms)./sumnorms;
    
end

