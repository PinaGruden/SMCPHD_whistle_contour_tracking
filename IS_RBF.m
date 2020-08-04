function [y_out] = IS_RBF(x_in,w,C,vari)
%IS_RBF is a function that uses Radial Basis Function (RBF) network to
%predict the next location of particles (frequency and chirp). The RBF
%function needs to be trained separately to obtain network centres,
%variances and weights. This one uses Gaussian RBFs.

% Inputs
% x_in = particle positions (freq and chirp) (N x dim)
% w = learned weights of RBFs (M+1 x dim)- should contain a weight for bias 
% factor as well (hence M+1)
% C = learned RBF centres locations (M x dim)
% vari = the learned variances of the centres (M x dim)

% Outputs
%y_out= predicted particle positions (freq and chirp) (N x dim)

% Pina Gruden, ISVR, May 2017

% Compute Gaussian RBFs:

N=size(x_in,1); %number of datapoints/particles
M=size(C,1); %number of RBF centres
Phi=zeros(N,M); %preallocate

for j=1:M
df=x_in-C(j,:);
R=diag(1./vari(j,:)); %compute the inverse of the  covariance matrix
df1=df*R;
% R=diag(vari(j,:)); %compute the inverse of the  covariance matrix
% df1=df/R;
dfm=sum(df1.*df,2);
Phi(:,j)=exp(-dfm./2); %compute gaussian pdf for each particle
end

Phi=[Phi,ones(N,1)]; %add a bias factor
y_out=Phi*w; % predicted particle locations (freq and chirp)

end

