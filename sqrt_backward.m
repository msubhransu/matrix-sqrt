function dldx = sqrt_backward(X, dldy, cache, method, maxIter)
switch lower(method)
 case 'matrix-backprop' % MATLAB's implementation (CPU only)
    dldx = matrix_backprop(X, cache{1}, cache{2}, dldy);
 case 'lyap-svd'   % sqrt via SVD
    dldx = lyap_svd(cache{1}, cache{2}, -dldy);
 case 'lyap-ns'    % sqrt via Denmen-Bevers iterations
    if nargin < 5
        maxIter = 10;
    end
    dldx = lyap_newton_shulz(X, dldy, maxIter);
end

function dldx = lyap_svd(U, D, C)
[ma, ~] = size(D);
D = sqrt(D);
IU = U';
D = D*ones(ma,ma);
X = -U*(IU*C*IU.'./(D+D.'))*U.';
dldx = real(X);


function dlda = lyap_newton_shulz(z, dldz, maxIter)
if nargin < 3
   maxIter = 20;
end
scale = sqrt(sum(sum(z.*z)));
a = z/scale;
q = dldz/scale;
I = 3*eye(size(z,1));
for iter = 1:maxIter
  q = (q*(I - a*a) - a'*(a'*q-q*a))/2;
  a = a*(I - a*a)/2;
end
dlda = q/2;



function dldx = matrix_backprop(X, U, D, dldy)
[~, dim] = size(X);

diagS = diag(D);
ind = diagS > dim*eps(max(diagS));
Dmin = sum(ind);

D = D(ind,ind); U = U(:,ind);
            
dldU = 2.*dldy*U*diag(sqrt(diag(D)));
dldD = 0.5*diag(1./(sqrt(diag(D))+eps))*U'*dldy*U;

K = 1./(diag(D)*ones(1,Dmin)-(diag(D)*ones(1,Dmin))'); K(eye(size(K,1))>0)=0;

tmp = (K'.*(U'*dldU)) + diag(diag(dldD));

dldx = U*tmp*U';
dldx = 0.5*(dldx + dldx');
