%%% Andrew Bernoff 12/6/2024
%%%
%%% This code does parameter continuation for Ising minimizers.
%%%
%%% The Ising model minimizer computation occurs in QuadMinIsing
%%% The assumption is that we are minimizing
%%%
%%%     H=sum(J_ij sigma_i sigma_j ,i=1..n,j=1..n)
%%%
%%%  Over Nr realizations. Note that QuadMinIsing at present returns many
%%%  potential local minimizers with possible duplication.
%%%
    clear all

    Nr=20; % Number of realizations

%%  Random plot voodoo
    clf
    set(groot,'defaultAxesTickLabelInterpreter','latex');  
    set(groot,'defaulttextinterpreter','latex');
    set(groot,'defaultLegendInterpreter','latex');

%%  Main loop
    Np=20;      %  Parameter samples
    Ngrid=10;   %  Gridpoints in standard pore
    
    theta_start = pi;
    theta_finish = pi/10;
    theta_width = theta_finish/2;
    thetas = linspace(theta_start,theta_finish,Np);

    three = true;

for ip = 1:Np

    target(1) =  thetas(ip); 
    target(2) = -thetas(ip);
    if three
        target(3)= 0 ;
        width(3) = theta_width;
        ngrid(3) = Ngrid;
    end

    ntargets=numel(target);
    %
    width(1) = theta_width;
    width(2) = theta_width;
    %
    ngrid(1) = Ngrid;
    ngrid(2) = Ngrid;


% Eventually should check for overlap here

% Use midpoint rule for targets
    sigma=[];

    for targ=1:ntargets;
        gridsize=width(targ)/ngrid(targ);
        grid=target(targ)-width(targ)/2-gridsize/2 +gridsize*[1:ngrid(targ)];
        sigma=[sigma grid];
    end

    numsigma=numel(sigma);

% Preallocate output matrix
    if ip==1
        sigmatheta=zeros(numsigma,Np);
        sigmamin=zeros(numsigma,Np);
    end

    sigmatheta(:,ip)=sigma;

% Now construct the coupling matrix

    [nx,ny]=meshgrid(sigma,sigma);

% Define the coupling function - remember we are minimizing so J(0,0)
% should be a minimum

%    Jfun =@(theta1,theta2) - cos(theta1-theta2);
    Jfun =@(theta1,theta2) exp(- 10*cos(theta1-theta2))-2;
% Now compute the coupling matrix

    Jij = Jfun(nx,ny);

% Call the ISing minimization routine
    sigmaout=QuadMinIsing(Jij,Nr);

% Compute the energy
    for r=1:Nr   %%% Need to eliminate (or parfor) this loop eventually
         E(r)= ((sigmaout(:,r))'*Jij* sigmaout(:,r))/(numsigma)^2;
    end
%
  [Emin,rmin]=min(E(r));
%  
  sigmamin(:,ip)=sigmaout(:,rmin);
end

% Graph the answers

figure(1)
clf
            title('Ising Model (continuous)')
            ylabel('Polar Angle', 'Interpreter','latex')
            xlabel('Spin Parameter')
            
hold on 
    green = [0, .7 0];
% for tplot = 1:ntargets  % Plot targets
%     x2 = [eps,Nr+1, Nr+1,eps];
%     inBetween = [target(tplot)-width(tplot)/2, target(tplot)-width(tplot)/2,...
%                  target(tplot)+width(tplot)/2,target(tplot)+width(tplot)/2];
%     fill(x2, inBetween, green, 'EdgeColor', 'none');
%     alpha(.4)
% end

% for splot =1:numsigma  %Plot spin points
%     plot([0,Nr+1],[sigma(splot),sigma(splot)],'-','Color',[0.2 0.5 0.9 0.2])
% end

for iplot = 1:Np
    % Plot spin locations
    plot(thetas(iplot)*ones(size(sigmatheta(:,iplot))),sigmatheta(:,iplot),'.r' )
    % Plot minimizers
    sigmaON= sigmatheta(sigmamin(:,iplot)>0,iplot);
    plot(thetas(iplot)*ones(size(sigmaON)),sigmaON,'+g' )
end
    
Ax = gca;
Ax.TickLabelInterpreter = 'latex';

yt = [-pi pi];
ytl = {'$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'};
ytv = linspace(min(yt), max(yt), numel(ytl));
set(Ax, 'YTick',ytv, 'YTickLabel',ytl)

ylim(yt);
xlim([min(theta_start,theta_finish) max(theta_start,theta_finish)])

% plot(cdf,y,'r')
% plot(cdftheory,y,'k-')
% plot(cdf,1-y,'b')
hold off

% figure(2) 
% clf
% hold on
% 
% title('Ising Model (continuous)')
%             ylabel('Polar Angle', 'Interpreter','latex')
%             xlabel('Spin trials')
% plot(xpts,E, '*')
% xlim([0 Nr+1])
% ylim([-1 0]);
% hold off


function  sigma=QuadMinIsing(J,Nr)
%%% Note this code is agnostic about the spin locations
%%% It returns a list of potential minimizers

     maxiter=1000;
     niter=0;

    [nx,ny]=size(J);

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





