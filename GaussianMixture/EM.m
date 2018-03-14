%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - function applies EM algorithm 
% - iterates until convergence given tolerance tol 
% - takes as input tol, data x, and intial pars
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [LL,mu,sigma,pi] = EM(eps,tol,x,mu,sigma,pi)

    iter = 1;
    maxiter   = 1000;
    LL(iter)  = loglikeli(mu,sigma,pi,x); 
    tol_check = inf;
    
    while tol<tol_check && iter < maxiter
        
        % incrementation;
        iter = iter+1;
        
        % "E" step; 
        gamma = E_step(mu,sigma,pi,x);
        
        % "M" step;
        [mu,sigma,pi] = M_step(gamma,x);
        
        % heuristic avoidance of singularities;
        eps_mat = eye(size(sigma,1))*eps;
        for i=1:size(sigma,3)
            tempmat = sigma(:,:,i);
            tempmat(abs(tempmat)<eps_mat)=eps_mat(abs(tempmat)<eps_mat);
            sigma(:,:,i) = tempmat;
        end
        
        % save and check;
        LL(iter) = loglikeli(mu,sigma,pi,x);
        tol_check = abs(LL(iter)-LL(iter-1));
        
    end
   
end
