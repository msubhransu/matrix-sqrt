% Set parameters for testing
n = 512;
tau = 1;
numPoints = 1000;
isFloat = true;
isGPU = true;
maxIter = 10;
runIter = 10;

fprintf('Experiment:\n')
fprintf('>>>> n %i, tau %.1f, numPoints %i, float %i, gpu %i, maxIter %i, runIter %i\n', ...
          n, tau, numPoints, isFloat, isGPU, maxIter, runIter);

% Generate points randomly to create a PSD matrix
X = randn(numPoints, n);
A = (X'*X)/size(X,1) + tau*eye(n);
dldz = rand(n);
dldz = 0.5*(dldz + dldz');

% Calculate quantities with full precision on the CPU
sqrtA_true = sqrtm(A);
dlda_true = lyap2(sqrtA_true, -dldz);

if isFloat
   A = single(A);
   dldz = single(dldz);
   fprintf('>>>> floating point: single\n');
else
   fprintf('>>>> floating point: double\n');
end

if isGPU
   A = gpuArray(A);
   dldz = gpuArray(dldz);
   fprintf('>>>> GPU: true\n');
else
   fprintf('>>>> GPU: false\n');
end

% Evaluate forward models
% SVD
fprintf('\nForward time:\n');
tic
for i = 1:runIter
    [sqrtA_svd, cache] = sqrt_forward(A, 'svd');
end
fprintf(' %fs SVD\n', toc);

% Denman Beavers
tic
for i = 1:runIter
    [sqrtA_db, ~] = sqrt_forward(A, 'db', maxIter);
end
fprintf(' %fs Denman-Beavers (%i iter)\n', toc, maxIter);

% Newton Schulz
tic
for i = 1:runIter
    [sqrtA_ns, ~] = sqrt_forward(A, 'ns', maxIter);
end
fprintf(' %fs Newton-Schulz (%i iter)\n', toc, maxIter);


% Evalute backward models
fprintf('\nBackward time:\n');
tic
for i = 1:runIter 
 dlda_mb = sqrt_backward(A, dldz, cache, 'matrix-backprop', maxIter);
end
fprintf(' %fs Matrix-backprop\n', toc);

tic
for i = 1:runIter 
 dlda_svd = sqrt_backward(A, dldz, cache, 'lyap-svd', maxIter);
end
fprintf(' %fs Lyapunov SVD\n', toc);

tic
for i = 1:runIter 
 dlda_ns = sqrt_backward(sqrtA_ns, dldz, cache, 'lyap-ns', maxIter);
end
fprintf(' %fs Lyapunov Newton-Schulz (%i iter)\n', toc, maxIter);

if isGPU
   sqrtA_svd = double(gather(sqrtA_svd));
   sqrtA_db = double(gather(sqrtA_db));
   sqrtA_ns = double(gather(sqrtA_ns));
   dlda_mb = double(gather(dlda_mb));
   dlda_svd = double(gather(dlda_svd));
   dlda_ns = double(gather(dlda_ns));
end

matrix_norm = sqrt(sum(sum(A.*A)));
matrix_norm_sqrt = sqrt(sum(sum(sqrtA_true.*sqrtA_true)));

% Forward errors
fprintf('\nForward error:\n');

error_fwd = sqrt(sum(sum((sqrtA_svd - sqrtA_true).*(sqrtA_svd - sqrtA_true))))/matrix_norm_sqrt;
fprintf(' %d SVD\n', error_fwd);

error_fwd = sqrt(sum(sum((sqrtA_db - sqrtA_true).*(sqrtA_db - sqrtA_true))))/matrix_norm_sqrt;
fprintf(' %d Denman-Beavers (%i iter)\n', error_fwd, maxIter);

error_fwd = sqrt(sum(sum((sqrtA_ns - sqrtA_true).*(sqrtA_ns - sqrtA_true))))/matrix_norm_sqrt;
fprintf(' %d Newton-Schulz (%i iter)\n', error_fwd, maxIter);

% Backward errors
fprintf('\nBackward error:\n');

error_bkwd = sqrt(sum(sum((dlda_mb - dlda_true).*(dlda_mb - dlda_true))))/matrix_norm;
fprintf(' %d Matrix-backprop\n', error_bkwd);

error_bkwd = sqrt(sum(sum((dlda_svd - dlda_true).*(dlda_svd - dlda_true))))/matrix_norm;
fprintf(' %d Lyapunov SVD\n', error_bkwd);

error_bkwd = sqrt(sum(sum((dlda_ns - dlda_true).*(dlda_ns - dlda_true))))/matrix_norm;
fprintf(' %d Lyapunov Newton-Schulz (%i iter)\n', error_bkwd, maxIter);
