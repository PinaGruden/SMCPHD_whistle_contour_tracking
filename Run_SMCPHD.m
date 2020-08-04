%% ~~~~~~~~~~~~ SMC-PHD for dolphin whistle tracking ~~~~~~~~~~~~~~~~~~~~~
clear

%This example runs the version of the SMC-PHD filter that uses the Radial
%Basis Function (RBF) network to draw persistent particles in the prediction step

%For details see Gruden, P. and White, P. (2020). Automated extraction of dolphin whistles - 
%a Sequential Monte Carlo Probability Hypothesis Density (SMC-PHD)
%approach; Journal of the Acoustical Society of America.

%//////////// LOAD sample Ground Truth and Measurement data //////////////

load('Measurement_and_GT_demo.mat');
% GT = ground truth data
% Zset = measurements (spectral peaks)
% dt = time step between consecutive windows (s)


%% //////////// INITIALIZE SMC-PHD PARAMETERS and MODELS /////////////
% %--------------- Parameters for SMCPHD ------------------------------------
% Table 1 in Gruden & White (2020):
parameters.pdet = 0.99; %probability of detection 
parameters.psurv = 0.994;%probability of survival 
parameters.Mp=50; %number of particles per persistent target
parameters.Nb=50; %number of particles per newborn target
parameters.nClutter = 10; % number of clutter points per time step
parameters.wth=0.0005; % threshold for state estimation (\eta) 

% Load GMM distribution for drawing the chirp - used in target birth step
% Eq. (9) in Gruden & White (2020):
load('gmm_chirp'); 
parameters.gmm_all=gmm_all;

% %-------------- Models for SMCPHD---------------:
models.H = [1,0]; % measurement matrix
models.R = round((fs/win_width)^2/12); % measurement noise variance for Eq. (5) in Gruden & White (2020)
load('birthpdf.mat'); %start frequency distribution for drawing adaptive 
% weight in Eq. (10) in Gruden & White (2020)
models.birthpdf=birthpdf;
models.dt=dt;
models.F = [1, dt; 0, 1]; % state transition matrix (used to resolve state
%estimate label conflicts in  track_labels.m)

%--------------- Parameters for RBF network--------------------------------
% Learned RBF network- centres, variances and weights (from training data)
% Eq. (7):
load('Net_60RBF.mat')
RBFnet.C=C; % (c_j)
RBFnet.w=w; % (w_j)
RBFnet.vari=vari; % (Q_j)
% Learned process noise (n_k) - Eq. (7):
models.Q = [39.2,0;0,7326]; 

%% /////////// RUN SMC-PHD filter /////////////////

%Get state estimates and their labels:
[Xk,XkTag] = SMCPHD_RBF_adaptivebirth(Zset,parameters,models,RBFnet);

% Collect estimated states into tracks (whistle contours) based on their labels:
Track  = track_labels(XkTag,Xk,models);

% Impose track length criteria to get detections that match specified criterion:
tl=10;%target length criteria
c=1;ind=[];
for l=1:size(Track,2)
    if numel(Track(l).time)>=tl
        ind(c)=l;
        c=c+1;
    else
        continue
    end
end
DT=Track(ind); % detected whistles

%% ////////// PLOT RESULTS ///////////////
figure,
subplot(211) %plot measurements
T=0:models.dt:size(Zset,2)*models.dt-models.dt; %time vector for plotting
for k=1:size(Zset,2)
plot(T(k),Zset{k},'k.'),hold on
end
ylabel('Frequency (Hz)')
xlabel('Time (s)')
xlim([0,T(end)])
title('Measurements')

subplot(212)
% Plot GT:
col= [0,0,0]+0.8;
for k=1:size(GT,2)
    if GT(k).valid==1
    plot(GT(k).time,GT(k).freq,'-','Color', col,'LineWidth',4),hold on
    else
    plot(GT(k).time,GT(k).freq,':','Color', col,'LineWidth',4),hold on    
    end
end

% Plot Detections:
for k=1:size(DT,2)
    plot(DT(k).time,DT(k).freq,'-','LineWidth',2),hold on
end

ylabel('Frequency (Hz)')
xlabel('Time (s)')
xlim([0,T(end)])
title('Ground truth data and SMC-PHD detections')
subplot(212), h1=plot(NaN,NaN,'-','Color', col,'LineWidth',4);
subplot(212), h2=plot(NaN,NaN,':','Color', col,'LineWidth',4);
subplot(212), h3=plot(NaN,NaN,'r-','LineWidth',2);
legend([h1,h2,h3],'Ground truth (valid)', 'Ground truth (not valid)','SMC-PHD detection')