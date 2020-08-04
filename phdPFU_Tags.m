function [ w_C, x_C,Tag_C, pe] = phdPFU_Tags(w_Cz,x_Cz,Tag_Cz,z,H,R,clutter,pdet,L)
%phdPFU_Tags is a PHD particle filter update according to Ristic et al.(2016).
%An Overview of Particle Methods for Random Finite Set Models. Inf.
% Fusion 31: 110-126.

%//////// Inputs: /////////
% w_Cz = particle weights corresponding to a particle cluster of a given
% measurement z_k
% x_Cz = particle states corresponding to a particle cluster of a given
% measurement z_k
% z = a given measurement z_k from the measurement set Z_k at time k
% H = measurement matrix
% R = measurement noise variance
% clutter = clutter PHD
% pdet = probability of detection
% L = user defined number of how many times the cluster gets resampled


%//////// Outputs: /////////
% w_C = resampled particle weights belonging to a cluster,
% x_C = resampled particle states belonging to a cluster,
% pe = probability of a cluster's existance 

%Pina Gruden, ISVR, 2017

M=length(w_Cz);
gk= zeros(1,M);

% perform particle update
for i=1:M
gk(i) = normpdf(z,H*x_Cz(:,i),sqrt(R)); %observation likelihood p(z_k|x_k)
end

denom = clutter + (pdet*w_Cz*gk');
w_C = pdet*(gk.*w_Cz)/denom;
pe = sum(w_C); %probability of the cluster existance 

% Resample
%Stratified resampling
indx  = resampleStratified(w_C./pe, L);
x_C = x_Cz(:,indx); %extract new particles
w_C= repmat(pe/L, 1, L); %all particles have the same weight1/N
Tag_C=Tag_Cz(:,indx); 

% here you could also apply MCMC move to diversify particles

end

