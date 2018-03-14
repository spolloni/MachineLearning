function E = Efun(w,X,y,C)
    
    % Quadratic term;
    QuadTerm = 0.5*sum(diag(w'*w));
    
    % Cross-products to evaluate L;
    Xw = X*w;
    Xw_y = (Xw.*dummyvar(y))*ones(size(dummyvar(y),2),1);
    Xw = Xw + ones(size(Xw));
    Xw(dummyvar(y)==1) = -Inf;
    [M,Ind] = max([Xw Xw_y],[],2);
    Ind = (Ind ~= 1+ size(dummyvar(y),2));
    M = max(Xw,[],2);
   
    % Loss term;
    LossTerm = sum(Ind.*(M-Xw_y));
   
    % Compute E;
    E = QuadTerm + C*LossTerm;
    
end