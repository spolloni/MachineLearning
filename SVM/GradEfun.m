function GradE = GradEfun(w,X,y,C)
    
    % Derivative of quadratic term;
    GradQuadTerm = w;
    
    % Cross-products to evaluate L;
    Xw = X*w;
    Xw_y = (Xw.*dummyvar(y))*ones(size(dummyvar(y),2),1);
    Xw = Xw + ones(size(Xw));
    Xw(dummyvar(y)==1) = -Inf;
    [M,Ind1] = max([Xw Xw_y],[],2);
    [M,Ind2] = max(Xw,[],2);

    % Fill-in derivatives based on results from Problem 1
    Y = permute(y,[2 3 1]);
    Y = repmat(Y,size(X,2),1,1);
    XX = permute(X,[2 3 1]);
    XX = repmat(XX,1,size(dummyvar(y),2),1);
    IInd1 = permute(Ind1,[2 3 1]);
    IInd1 = repmat(IInd1,size(X,2),size(dummyvar(y),2),1);
    IInd2 = permute(Ind2,[2 3 1]);
    IInd2 = repmat(IInd2,size(X,2),1,1);
    XX(IInd1 == 1+ size(dummyvar(y),2)) = 0;
    for i=1:size(dummyvar(y),2);
      temp = XX(:,i,:);
      temp(temp~=0 & Y==i) = -temp(temp~=0 & Y==i);
      temp(temp~=0 & Y~=i & IInd2~=i)=0;
      XX(:,i,:) = temp;
    end
    
    % Derivative of Loss term;
    GradLossTerm = sum(XX,3);
   
    % Compute gradient of E;
    GradE = GradQuadTerm + C*GradLossTerm;
    
end