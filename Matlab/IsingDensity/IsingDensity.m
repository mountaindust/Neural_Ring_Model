%%% 
%%% 
%%% This code is a general minimization code for a continuous Ising Model
%%% 
%%% The assumption is that we are minimizing
%%%
%%%     H=sum(J_ij sigma_i sigma_j ,i=1..n,j=1..n)
%%%
%%%  Over Nr realizations. Note that QuadMinIsing at present returns many
%%%  potential local minimizers with possible duplication.
%%%
   clear all
   close all

%%  Random plot voodoo
    clf
    set(groot,'defaultAxesTickLabelInterpreter','latex');  
    set(groot,'defaulttextinterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');
%
    beta = 0.1; % Boltzman parameter
%% create thetas for Ising - assume theta is in [-pi,pi]
    theta(1) =  pi/3; 
    theta(2) = -pi/3;
    K=numel(theta);
    [thetaj,thetak]=meshgrid(theta,theta);

% Define the coupling function - remember we are minimizing so J(0,0)
% should be a minimum

    Jfun =@(theta1,theta2) - cos(theta1-theta2);

% Now compute the coupling matrix

    J=Jfun(thetaj,thetak);

%%  Now use number of neurons to get N, rho
    n(1) = 100;
    n(2) = 200;
    %
    N=sum(n);
    rho = n/N;
    betaN=N*beta;

% Call the Ising minimization routine
    Isingout=GlobalMinIsing(betaN,Jij,rho);

% Compute the energy
    for r=1:Nr   %%% Need to eliminate (or parfor) this loop eventually
         E(r)= ((sigmaout(:,r))'*Jij* sigmaout(:,r))/(numsigma)^2;
    end

    [ max(E),min(E)]

% Graph the answers
    xpts=[1:Nr];


figure(1)
clf
            title('Ising Model (continuous)')
            ylabel('Polar Angle', 'Interpreter','latex')
            xlabel('Spin trials')
            
hold on 
    green = [0, .7 0];
for tplot = 1:ntargets
    x2 = [eps,Nr+1, Nr+1,eps];
    inBetween = [target(tplot)-width(tplot)/2, target(tplot)-width(tplot)/2,...
                 target(tplot)+width(tplot)/2,target(tplot)+width(tplot)/2];
    fill(x2, inBetween, green, 'EdgeColor', 'none');
    alpha(.4)
end

for splot =1:numsigma
    plot([0,Nr+1],[sigma(splot),sigma(splot)],'-','Color',[0.2 0.5 0.9 0.2])
end

for nrplot = 1:Nr
    sigmaON= sigma(sigmaout(:,nrplot)>0);
    plot(nrplot*ones(size(sigmaON)),sigmaON,'+k' )
end
    
Ax = gca;
Ax.TickLabelInterpreter = 'latex';

yt = [-pi pi];
ytl = {'$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'};
ytv = linspace(min(yt), max(yt), numel(ytl));
set(Ax, 'YTick',ytv, 'YTickLabel',ytl)

ylim(yt);
xlim([0 Nr+1])

% plot(cdf,y,'r')
% plot(cdftheory,y,'k-')
% plot(cdf,1-y,'b')
hold off

figure(2) 
clf
hold on

title('Ising Model (continuous)')
            ylabel('Polar Angle', 'Interpreter','latex')
            xlabel('Spin trials')
plot(xpts,E, '*')
xlim([0 Nr+1])
ylim([-1 0]);
hold off


function  sigma=GlobalMinIsing(betaN,Jij,rho)

     maxiter=1000;
     niter=0;

 %%% Initial Guess is random binary matrix

    sigma = randi([0,1],nx,Nr);

    badlist=[1:Nr]; %Initial rows that are not minimizers

    while(~isempty(badlist) )
        %%% Need to compute the effect of changing each sigma
        %%% Basic Formula:
        %%%     dJ = (1-2*sigma(k))*(R(k)+L(k)) + D(k)
        %%%     D(k)= diag(J), R(k) =J*sigma(k) L(k) = sigma(k)*J
        %%%     Paying attention to rows and columns               
             D =diag(J(:,:));
        for r=1:Nr   %%% Need to eliminate (or parfor) this loop eventually
          R = J(:,:)*sigma(:,r);
          L = (J(:,:))'*sigma(:,r);
          dJ(:,r) = (1-2*sigma(:,r)).*(R+L)+D;
        end
        %%%
          down = dJ <0; % bitflips which reduce energy
          numbad = sum(down,1);
          badlist =find(numbad>0);
        %%% Need to eliminate (or parfor) this loop eventually
            for j=badlist
                idx = find(down(:,j)==1);
                n = numel(idx);
                in = randi(n);
                sigma(idx(in),j) =1-sigma(idx(in),j);
            end
        end


        %%% Finally, increase iterations and see if we have exceeded maximum  
        niter=niter+1;
        if(niter==maxiter)
             disp(['Maximum Iterations Exceeded'])
             badlist=[];  % This will cause us to exit the while loop
        end
    end





