%%%%%%%%%%%%%%%%%%
% Exercise on SVM
% Stefano Polloni 
%

clear;
clc;

load('digits');
y=[];
X=[];
Xtest=[];

%stack train and test data;
for i=1:10
    eval([ 'X = [X ; train' num2str(i-1) '];' ]);
    eval([ 'Xtest = [Xtest ; test' num2str(i-1) '];' ]);
    y = [y ; i*ones(500,1)];
end

% augment examples with constant feature 1;
Xtest = [Xtest ones(5000,1)];
X     = [X ones(5000,1)];

%set algorithm parameters;
T  = 500; 
r  = 0.01;

% matrice containing obj function values;
EE = zeros(8,T+1);

% matrice containing parameters w;
W  = zeros(785,10,8);

% Gradient descent part A;
for j = -4:3
    
    %initialize
    C = 10^j;
    w = zeros(785,10); 

    EE(j+5,1) = Efun(w,X,y,C);

    for i=1:T
    
        % compute gradient;
        grad = GradEfun(w,X,y,C);
    
        % update parameter;
        w = w  - r*grad ;
    
        % compute objective function;
        EE(j+5,i+1) = Efun(w,X,y,C);
    
        % Adjust learning rate;
        Adj = ((EE(j+5,i)-EE(j+5,i+1))/EE(j+5,i))+1;
        if Adj>0 && Adj<2
            r = Adj*r;
        end    
        
        % Display stuff to get sense of progress;
        [ C i EE(j+5,i+1) r ]
           
    end  

    W(:,:,j+5)= w;  

end

% Plot for part B;
figure(2);
plot([1:1:T+1],EE(5,:),'k','LineWidth',1.5);
xlabel('Number of Iterations');
ylabel('Value of Objective Function');
print('ObjFun', '-dpng', '-r300'); 

% Classification rates for part C;
CC = zeros(1,8);
ConfMat = zeros(10,10,8);
for i=1:8
    
    %count correctly classified digits;
    [M,pred_class] = max(Xtest*W(:,:,i),[],2);
    correct = (pred_class==y);
    CC(i) = sum(correct);
    
    %compute classification matrix;
    predclass = reshape(pred_class,500,10);
    for j=1:10     
        tab = histc(predclass(:,j),1:10);
        ConfMat(j,:,i) = tab';
    end
    
end

% Plots for part D;
W = W(1:end-1,:,:);
[m,ind] = max(CC);
digits = permute(reshape(W(:,:,ind),28,28,10),[2 1 3]);
figure(3);
for i=1:10
    subplot(4,3,i);
    colormap(flipud(gray));
    imagesc(digits(:,:,i));
end
print('digits', '-dpng', '-r300'); 
