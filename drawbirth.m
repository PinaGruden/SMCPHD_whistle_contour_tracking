function [x] =drawbirth(m,P,N)
%draw N samples from distribution with mean m and covariance P

n=length(m);             % get the dimension
C=chol(P);               % perform cholesky decomp R = C'C
X=randn(n,N);
x=C'*X+m*ones(1,N);
end