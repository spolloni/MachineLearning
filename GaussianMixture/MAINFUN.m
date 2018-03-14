%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise on Gaussian Mixtures
% Stefano Polloni 
%

clear;
clc;

%load data;
load data2;
x2 = data;
load data3;
x3 = data;

% draw d random assigments to clusters;
d = 2000;
draw2 = randi([1 2],size(x2,1),d);
draw3 = randi([1 3],size(x3,1),d);

% set tolerance for convergence;
tol = .00001;

% set epsilon for singularities heuristic;
eps = 0.02;

% MIXTURE OF 2 GAUSSIANS

for i=1:d
% loop through initializations
    
    %dummy matrix for computation;
    dum = dummyvar(draw2(:,i));
    
    %initialize pi;
    pi = (sum(dum)/size(dum,1))';
    
    %initialize means;
    mu = (x2'*dum)./repmat(sum(dum),2,1);
    
    %initialize covariance;
    sigma(:,:,1) = cov(x2(dum(:,1)>0,:));
    sigma(:,:,2) = cov(x2(dum(:,2)>0,:));
    
    %run EM algo given eps, tol and initialization;
    [LL,mu,sigma,pi] = EM(eps,tol,x2,mu,sigma,pi);
    
    MU{i} = mu;
    SIGMA{i} = sigma;
    PI{i} = pi;
    
    LLL{i} =LL;
    LLend(i) = LL(end);
       
end

% indice of best final likelihood;
[~,i] = max(LLend);

% plot of "best run";
figure(1);
plot([1:1:size(LLL{i},2)],LLL{i},'k','LineWidth',1.5);
xlabel('Number of Iterations');
ylabel('Log-Likelihood');
print('gaussian2_LL', '-dpng', '-r300');
close(figure(1));

% parameters of "best run";
mu = MU{i}
sigma = SIGMA{i}
pi = PI{i}

%visualization;
x = -10:.1:20; 
y = -10:.1:20;
[X,Y] = meshgrid(x,y);
F1 = mvnpdf([X(:) Y(:)],mu(:,1)',sigma(:,:,1));
F1 = reshape(F1,length(y),length(x));
F2 = mvnpdf([X(:) Y(:)],mu(:,2)',sigma(:,:,2));
F2 = reshape(F2,length(y),length(x));
figure(2);
%scatter(x2(:,1),x2(:,2),5,'k','o','LineWidth',.7);
scatter(x2(:,1),x2(:,2),15,'k');
hold on;
contour(x,y,F1,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999],'k-');
hold on;
contour(x,y,F2,[.0001 .001 .01 .05:.01:.95 .99 .999 .9999],'k-');
hold on;
scatter(mu(1,1),mu(2,1),'o','LineWidth',.7,'MarkerEdgeColor','k','MarkerFaceColor','r')
hold on;
scatter(mu(1,2),mu(2,2),'o','LineWidth',.7,'MarkerEdgeColor','k','MarkerFaceColor','r')
hold off;
xlabel('x1'); ylabel('x2');
print('gaussian2', '-dpng', '-r300');
close(figure(2));

% MIXTURE OF 3 GAUSSIANS

for i=1:d
% loop through initializations
    
    %dummy matrix for computation;
    dum = dummyvar(draw3(:,i));
    
    %initialize pi;
    pi = (sum(dum)/size(dum,1))';
    
    %initialize means;
    mu = (x3'*dum)./repmat(sum(dum),2   ,1);
    
    %initialize covariance;
    sigma(:,:,1) = cov(x3(dum(:,1)>0,:));
    sigma(:,:,2) = cov(x3(dum(:,2)>0,:));
    sigma(:,:,3) = cov(x3(dum(:,3)>0,:));
    
    %run EM algo given eps, tol and initialization;
    [LL,mu,sigma,pi] = EM(eps,tol,x3,mu,sigma,pi);
    
    MU{i} = mu;
    SIGMA{i} = sigma;
    PI{i} = pi;
    
    LLL{i} =LL;
    LLend(i) = LL(end);
       
end

% indice of best final liklihood;
[~,i] = max(LLend);

% plot of "best run";
figure(3);
plot([1:1:size(LLL{i},2)],LLL{i},'k','LineWidth',1.5);
xlabel('Number of Iterations');
ylabel('Log-Likelihood');
print('gaussian3_LL', '-dpng', '-r300');
close(figure(3));

% parameters of "best run";
mu = MU{i}
sigma = SIGMA{i}
pi = PI{i}

%visualization;
x = -10:.1:15; 
y = -10:.1:20;
[X,Y] = meshgrid(x,y);
F1 = mvnpdf([X(:) Y(:)],mu(:,1)',sigma(:,:,1));
F1 = reshape(F1,length(y),length(x));
F2 = mvnpdf([X(:) Y(:)],mu(:,2)',sigma(:,:,2));
F2 = reshape(F2,length(y),length(x));
F3 = mvnpdf([X(:) Y(:)],mu(:,3)',sigma(:,:,3));
F3 = reshape(F3,length(y),length(x));
figure(4);
scatter(x3(:,1),x3(:,2),15,'k');
hold on;
contour(x,y,F1,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999],'k-');
hold on;
contour(x,y,F2,[.0001 .001 .01 .05:.01:.95 .99 .999 .9999],'k-');
hold on;
contour(x,y,F3,[.0001 .001 .01 .05:.01:.95 .99 .999 .9999],'k-');
hold on;
scatter(mu(1,1),mu(2,1),'o','LineWidth',.7,'MarkerEdgeColor','k','MarkerFaceColor','r')
hold on;
scatter(mu(1,2),mu(2,2),'o','LineWidth',.7,'MarkerEdgeColor','k','MarkerFaceColor','r')
hold on;
scatter(mu(1,3),mu(2,3),'o','LineWidth',.7,'MarkerEdgeColor','k','MarkerFaceColor','r')
hold off;
xlabel('x1'); ylabel('x2');
print('gaussian3', '-dpng', '-r300');
close(figure(4));


