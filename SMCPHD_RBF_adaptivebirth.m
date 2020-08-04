function [Xk,XkTag,weights,states] = SMCPHD_RBF_adaptivebirth(Zset,parameters,models,RBFnet)
%SMCPHD_RBF_adaptivebirth is a function that tracks dolphin whistle
%contours with the SMC-PHD filter. The details are given in our
%publication:
% Gruden, P. and White, P. (2020). Automated extraction of dolphin whistles - 
%a Sequential Monte Carlo Probability Hypothesis Density (SMC-PHD)
%approach; Journal of the Acoustical Society of America.

% The implementation is based on the paper by Ristic, B. et al. (2016).
% An Overview of Particle Methods for Random Finite Set Models. Information
% Fusion 31: 110-126.


%Pina Gruden, Institute of Sound and Vibration Research (ISVR), University
%of Southampton, UK; and Research Corporation of the University of Hawaii
%(RCUH), University of Hawaii, US.

%---------- INPUTS: -----------
% ~ Zset = measurement set; cell array of dimension 1 x T, where T = number
%          of time steps. 
% ~ parameters = parameters for the SMCPHD filter; a struct containing
%               fields:
%               - parameters.pdet; %probability of target detection
%               - parameters.psurv; %probability of target survival 
%               - parameters.Mp; %number of particles per persistent target
%               - parameters.Nb; %number of particles per newborn target
%               - parameters.nClutter; %number of clutter points per time step
%               - parameters.wth; % threshold for State estimation
%               - parameters.gmm_all; %prior for drawing chirp of the newborn targets 
% ~ models = models for the SMCPHD filter; a struct containing fields:
%           - models.Q; % system noise covariance matrix 
%           - models.H; % measurement matrix
%           - models.R; % measurement noise covariance matrix 
%           - models.birthpdf; %a prior for drawing adaptive weight for
%           newborn particles
% ~ RBFnet = trained Radial Basis Function (RBF) network; a struct
%           containing fields:
%           - RBFnet.C; %RBF net centres
%           - RBFnet.w; %RBF net weights
%           - RBFnet.vari; %RBF net centre variances

%--------- OUTPUTS: ----------
% ~ Xk = state estimates for each time k; cell array of dimension 1 x T, 
%       where T= number of time steps.
% ~ XkTag = identities (tags) of the estimated states; cell array of 
%           dimension 1 x T.
% ~ weights = particle weights for each time k; 1 x T cell array
% ~ states = particles for each time k; 1 x T cell array


T=size(Zset,2);
%get the filter parameters
pdet = parameters.pdet; %probability of target detection 
psurv = parameters.psurv; %probability of target survival 
Mp=parameters.Mp; %number of particles per persistent target
Nb=parameters.Nb; %number of particles per newborn target
clutter=parameters.nClutter/(50000-2000);
wth=parameters.wth; % threshold for State estimation
gmm_all=parameters.gmm_all; %prior for drawing chirp of the newborn targets

%get the system and observation model parameters
Q = models.Q; % noise covariance matrix for the system noise 
H = models.H; % measurement matrix
R = models.R; % noise covariance matrix for the measurement noise
%get the RBF net parameters (learned from training data)
C= RBFnet.C; %RBF net centres
wRBF = RBFnet.w; %RBF net weights
vari = RBFnet.vari; %RBF net centre variances
birthpdf = models.birthpdf;

%pre-allocate
Lk= zeros(1,T);weights=cell(1,T);states=weights;
Xk=cell(1,T);
XkTag=cell(1,T);Tags=cell(1,T);

for k=1:T

%% /////////////////////  Initialization  ////////////////////////////////
if k==1 %for the first time step
    N0=1; %Initial number of targets
    L0=100; %initial number of particles per target
    Lk_1=L0*N0;
    wk_1=repmat(N0/L0,1,Lk_1);
    xk_1=repmat([10000;0],1,L0);
    Tag=zeros(1,Lk_1);%Assign unique tag to each particle
    rmax=0;
else
    Lk_1 = Lk(k-1); %number of particles at previous step (JOINED particles from persistent and newborn)
    wk_1= weights{k-1}; %weights from previous step
    xk_1= states{k-1}; %states from previous step
    Tag=Tags{k-1};
end


if isempty(Zset{k}) %If there are no measurements then all targets are missdetected
    if ~isempty(wk_1)
        %% /////////////////////////  Prediction  /////////////////////////////
        %draw particles from importance distribution
        xk_1=(IS_RBF(xk_1',wRBF,C,vari))' + Q*randn(2,Lk_1);
        %compute weights
        wk_1=psurv.*wk_1;
        %% /////////////////////////// Update  ////////////////////////////////
        wk_1=wk_1.*(1-pdet);
        wki=find(wk_1==0);
        wk_1(wki)=[];
        xk_1(:,wki)=[];
        Tag(:,wki)=[];
        
        states{k} = xk_1;
        weights{k}=wk_1;
        Lk(k)=length(wk_1);
        Tags{k}=Tag;
    else
        states{k} = [];
        weights{k}=[];
        Lk(k)=0;
        Tags{k}=[];
    end
else %There are measurements available  at present time step
    Zk=Zset{k}(1,:);
    if ~isempty(wk_1)    %there are particles from prev step
    %% /////////////////////////  Prediction  /////////////////////////////
    %------------Persistent--------------
    %draw particles from importance distribution
    xk_1=(IS_RBF(xk_1',wRBF,C,vari))'+ Q*randn(2,Lk_1);
    %compute weights
    wk_1=psurv.*wk_1;
    
    %% //////////////////// Update and Partition particles //////////////////
    
     %--------------Persistent particles------------
    
    %~~~~~~~~~~~~~~~Compute the Pseudo-Likelihood~~~~~~~~~~~~~~~~~~~
    %for all particles given the measurements (Eq. 50 - Ristic et al., 2016)
    mk=size(Zk,2); %number of measurements
    Hxk_1=H*xk_1;
    gk=zeros(Lk_1,mk); %gk(j,n) where j is number of particles
    %and n is number of measurements
   
    %Compute Gaussian pdf - since it's all 1D (mesurements only contain
    %freq info) use the 1d pdf- this will have to be changed to the mvnpdf
    %if the meaurements should contain more features
    term1 = 1/(sqrt(R)*sqrt(2*pi)); %controls the height of the Gaussian
    for n=1:mk %iterate through measurements
        difer = Hxk_1 - Zk(:,n);
         gk(:,n)= term1*exp(-0.5*difer.^2./R);
    end
    
    Dz = (wk_1*gk).*pdet;
    denom= clutter+ Dz;
    %Pseudo-likelihood for each particle-measurement pair
    wks=((wk_1'.*gk)*pdet)./denom;
    %add the pseudolikelihood for the missed detections to wks (Eq. 51- Ristic et al., 2016):
    wks(:,mk+1)=wk_1'.*(1-pdet);
    
    %~~~~~~~~~~~~~~~~~~~~~Partition to Clusters ~~~~~~~~~~~~~~~~~~~~~~~~
    %compute probability distribution (p), based on wks, and draw cluster index
    %(indz),based on p, for each particle - so that all particles get
    %partitioned to a measurement or a missdetection cluster
  
    p=wks./sum(wks,2); % (Eq. 52 - Ristic et al., 2016)
    pind = find(isnan(p));
    p(pind,1:mk)=0;
    p(pind,mk+1)=1;
    indz=zeros(1,Lk_1);
    for j=1:Lk_1
        pcum=cumsum(p(j,:));
        rv=rand;
        indz(j)=find(rv<pcum,1,'first'); 
    end
    
    %~~~~~~~~~~~~~~~~~~~~ Mis-detection (mk+1) cluster ~~~~~~~~~~~~~~~~~~~~~~~
    %Treat and update separately
    w_Cz = wk_1(indz==mk+1);
    x_Cz = xk_1(:,indz==mk+1);
    Tag_Cz =  Tag(indz==mk+1);
    %find which of the particles are above a threshold:
    indmiss=find(w_Cz>=(1-pdet)*100/sum(indz==mk+1));
    
    %pre-allocate updated wight and state vectors
    z=1:mk;
    Cz =  z((ismember(1:mk,indz))); %which measurements form clusters
    nc=numel(Cz); %how many clusters there are based on
    %measuremetns (this excludes the miss detection cluster)
    L= Mp; %how many times each cluster gets resampled
    wk=zeros(1,nc*L + numel(indmiss));
    xk=zeros(2,nc*L + numel(indmiss));
    Tagk=zeros(1,nc*L + numel(indmiss));
    
    wk(end-numel(indmiss)+1:end)= w_Cz(indmiss).*(1-pdet);
    xk(:,end-numel(indmiss)+1:end)= x_Cz(:,indmiss);
    Tagk(:,end-numel(indmiss)+1:end)= Tag_Cz(indmiss);
    
    %~~~~~~~~~~~ Measurement clusters - Perform Update and Resample ~~~~~~~~~~~
      
    %pre-allocate:
    pez=zeros(1,nc);
    Cid=pez;
    ipez=1;
    indncs=1;
    indncst=L;
    
    for z=1:mk
        if any(indz == z)
            %perform update on the particles corresponding to the specific
            %cluster C(z)
            w_Cz = wk_1(indz==z);
            x_Cz = xk_1(:,indz==z);
            Tag_Cz = Tag(indz==z);
            [ w_C, x_C, Tag_C,pe] = phdPFU_Tags(w_Cz,x_Cz,Tag_Cz,Zk(:,z),...
                H,R,clutter,pdet,L); %updated and resampled weights (w_C) 
            %and states (x_C) for particles belonging to a cluster. pe is sum of w_C
            wk(indncs:indncst)=w_C;
            xk(:,indncs:indncst)=x_C;
            
            %find cluster identifier based on the max weight 
            [a,~,c] = unique(Tag_C');
            out = [a, accumarray(c,w_C')];
            [~,i]=max(out(:,2));
            id=out(i,1);
            
            if id == 0
            id = rmax+1;
            rmax =rmax+1;
            %change particle indices with ID 0 to the new id
            Tag_C(Tag_C==0) = id;
            end
            
            Tagk(indncs:indncst)=Tag_C;
            Cid(ipez)=id; %cluster ID
            pez(ipez)=pe; %cluster probability of existance
            ipez=ipez+1;
            indncs =indncst+1;
            indncst=indncs+(L-1);
        else
            continue
        end
    end
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~ State Estimation ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    %Perform if probability of a cluster is bigger than a threshold
    xz= Cz(pez>wth); %which measurements contribute to state estimates
    xk_est2 = zeros(2,numel(xz));ID=zeros(1,numel(xz));
    indncs=1;
    indncst=L;
    Tno2=0;
    for n=1:length(pez)
        w_C=wk(indncs:indncst);
        x_C=xk(:,indncs:indncst);
        indncs =indncst+1;
        indncst=indncs+(L-1);
        if pez(n)>wth
            Tno2=Tno2+1;
            xk_est2(:,Tno2)=x_C*w_C'./pez(n);
            ID(:,Tno2) = Cid(n); 
        end
    end
    
    Xk{k}=xk_est2;
    XkTag{k}=ID;
    
    %% //////////////////////  Target Birth ////////////////////////////////
    %Draw Nb newborn particles for each measurement that is not associated to
    %persistent target (so measurement not assiciated to Estimated States)
    
    zbirth = Zk(:,~ismember(1:mk,xz)); %measurements that can give birth to new targets
    nzb = numel(zbirth); %number of measurements creating Nb newborn particles
    
    %pre-allocate
    xk_b =zeros(2,Nb*nzb);
     w_birth =zeros(1,Nb*nzb);
    indncs=1;
    indncst=Nb;
    %draw particle states
    for j=1:nzb
        xk_b(1,indncs:indncst)=  drawbirth(zbirth(:,j),R,Nb);
        xk_b(2,indncs:indncst)= random(gmm_all,Nb);
        %to use adaptive target birth magnitude:
        v=birthpdf(1,:)-zbirth(:,j);
        [~,l]=min(abs(v));%index of the frequency that is closest to the measurement
        val=birthpdf(2,l);
        w_birth(1,indncs:indncst)=repmat(val/Nb,1,Nb);
        indncs =indncst+1;
        indncst=indncs+(Nb-1);
    end

    T_birth=zeros(1,Nb*nzb); %all newborn have ID 0;
   
    
    %% ////////////////  Combine Persistent and Newborn  ////////////////////
    % Save weights and states at current time step
    Lk(k)= length(wk) + length(w_birth); %number of particles at current step
    weights{k}=[wk,w_birth];
    states{k}=[xk,xk_b];
    Tags{k}=[Tagk,T_birth];
  else %there are no persisten particles, just initiate birth
            %% //////////////////////  Target Birth ////////////////////////////////
            %Draw Nb newborn particles for each measurement 
            zbirth = Zk; %measurements that can give birth to new targets
            nzb = numel(zbirth); %number of measurements creating Nb newborn particles
            
            %pre-allocate
            xk_b =zeros(2,Nb*nzb);
            w_birth =zeros(1,Nb*nzb);
            indncs=1;
            indncst=Nb;
            %draw particle states
            for j=1:nzb
                xk_b(1,indncs:indncst)=  drawbirth(zbirth(:,j),R,Nb);
                xk_b(2,indncs:indncst)= random(gmm_all,Nb);
                %to use adaptive target birth magnitude (like in GMPHD):
                v=birthpdf(1,:)-zbirth(:,j);
                [~,l]=min(abs(v));%index of the frequency that is closest to the measurement
                val=birthpdf(2,l);
                w_birth(1,indncs:indncst)=repmat(val/Nb,1,Nb);
                
                indncs =indncst+1;
                indncst=indncs+(Nb-1);
            end
         
            T_birth=zeros(1,Nb*nzb); %all newborn have ID 0;
            
            
            %% ////////////////  Combine Persistent and Newborn  ////////////////////
            % Save weights and states at current time step
            Lk(k)= length(w_birth); %number of particles at current step
            weights{k}=w_birth;
            states{k}=xk_b;
            Tags{k}=T_birth;
   end  
    
end
end

end

