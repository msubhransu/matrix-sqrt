function [sA, cache] = sqrt_forward(A, method, maxIter)
cache = {};
switch lower(method)
 case 'sqrtm' % MATLAB's implementation (CPU only)
   sA = sqrtm(A);
 case 'svd'   % sqrt via SVD
   [U, D] = eig(A);
   sA = U*diag(sqrt(diag(D)))*U';
   cache = {U, D};
 case 'db'    % sqrt via Denmen-Bevers iterations
   if nargin < 3
        maxIter = 10;
   end
   sA = sqrt_db(A, maxIter);
 case 'ns'    % sqrt via Newton-Schulz iterations
   if nargin < 3
     maxIter = 10;
   end
   sA = sqrt_ns(A, maxIter);
end

function sA = sqrt_db(A, maxIter)
Y = A; 
Z = eye(size(A,1));
for iter = 1:maxIter
  Y_ = (Y + inv(Z))/2;
  Z = (Z + inv(Y))/2;
  Y = Y_;
end
sA = Y;

function sA = sqrt_ns(A, maxIter)
scale = sqrt(sum(sum(A.*A)));
Y = A/scale; 
Z = eye(size(A,1));
I = 3*eye(size(A,1));
for iter = 1:maxIter
  T = (I - Z*Y)/2; 
  Y = Y*T;
  Z = T*Z;
end
sA = Y*sqrt(scale);