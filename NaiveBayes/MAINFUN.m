%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise on Naive Bayes Classification
% Stefano Polloni 
%

clear;
clc;
load('digits');

%stack test matrices into 3D array;
for i=1:10
    eval([ 'test(:,:,i) = test' num2str(i-1) ';' ]);
    eval([ 'train(:,:,i) = train' num2str(i-1) ';' ]);
end

%get ML estimates of u;
uML = permute(sum(train,1)./size(train,1),[2 3 1]);

%make visualizations;
uML_mat = permute(reshape(uML,28,28,10),[2 1 3]);
figure;
for i=1:10
    subplot(4,3,i);
    colormap(flipud(gray))
    imagesc(uML_mat(:,:,i));
end
print('digits', '-dpng', '-r300'); 

%make confusion matrix;
conf_mat = zeros(10,10);
one=ones(size(test));
uML_3D = repmat(permute(uML,[3 1 2]),500,1,1);
for i=1:10
    dup_test = repmat(test(:,:,i),1,1,10);
    log_p_xi_cond_y = dup_test.*log(uML_3D)+(one - dup_test).*log(one - uML_3D);
    log_p_xi_cond_y(dup_test==0 & uML_3D==0) = 0;
    log_p_xi_cond_y(dup_test==1 & uML_3D==0) = -inf;
    log_p_x_cond_y  = permute(sum(log_p_xi_cond_y,2),[3 1 2]);
    [max,maxind] = max(log_p_x_cond_y);
    tab = histc(maxind,1:10);
    conf_mat(i,:)=tab;
    clear max maxind;
end

%output confusion matrix;
conf_mat
sum(diag(conf_mat))


