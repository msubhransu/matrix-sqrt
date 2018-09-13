# Matrix square root and its gradient on the GPU
# Author: Subhransu Maji (smaji@cs.umass.edu)
# Date: Dec 19, 2017
import argparse
import torch
import numpy as np
import time as tm
from torch.autograd import Variable

# Compute error
def compute_error(A, sA):
  normA = torch.sqrt(torch.sum(torch.sum(A * A, dim=1),dim=1))
  error = A - torch.bmm(sA, sA)
  error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
  return torch.mean(error)

# Forward + Backward via SVD decomposition
def sqrt_svd_lyap(A, dldz, dtype):
  batchSize = A.data.shape[0]
  dim = A.data.shape[1]
  dlda = torch.zeros(batchSize, dim, dim).type(dtype)
  sA = torch.zeros(batchSize, dim, dim).type(dtype)
  for i in range(batchSize):
    U, S, V = (A[i,:,:].data).svd()
    sA[i,:,:] = (U.mm(S.diag().sqrt())).mm(V.t())
    S = S.diag().sqrt().mm(torch.ones(dim, dim).type(dtype))
    IU = U.t()
    X = -U.mm(
            ((IU.mm(dldz[i,:,:].data)).mm(IU.t()))
            /(S+S.t())
            ).mm(U.t());
    dlda[i,:,:] = X
  return sA, dlda, compute_error(A, Variable(sA, requires_grad=False))

# Forward via Denman-Beavers iterations
def sqrt_denman_beavers(A, numIters, dtype):
  batchSize = A.data.shape[0]
  dim = A.data.shape[1]
  sA = torch.zeros(batchSize, dim, dim).type(dtype)
  for n in range(batchSize):
    Y = (A[n,:,:]).data
    Z = torch.eye(dim, dim).type(dtype)
    for i in range(numIters):
      Y_ = 0.5*(Y + Z.inverse())
      Z = 0.5*(Z + Y.inverse())
      Y = Y_
    sA[n,:,:] = Y
  sA = Variable(sA, requires_grad=False)
  error = compute_error(A,sA)
  return sA, error

# Forward via Newton-Schulz iterations
# Backward via autograd
def sqrt_newton_schulz_autograd(A, numIters, dtype):
  batchSize = A.data.shape[0]
  dim = A.data.shape[1]
  normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
  Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
  I = Variable(torch.eye(dim,dim).view(1, dim, dim).
               repeat(batchSize,1,1).type(dtype),requires_grad=False)
  Z = Variable(torch.eye(dim,dim).view(1, dim, dim).
               repeat(batchSize,1,1).type(dtype),requires_grad=False)
  
  for i in range(numIters):
    T = 0.5*(3.0*I - Z.bmm(Y))
    Y = Y.bmm(T)
    Z = T.bmm(Z)
  sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
  error = compute_error(A, sA)
  return sA, error

# Forward via Newton-Schulz iterations (non autograd version)
# Seems to be slighlty faster and has much lower memory overhead
def sqrt_newton_schulz(A, numIters, dtype):
  batchSize = A.shape[0]
  dim = A.shape[1]
  normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
  Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
  I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
  Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
  for i in range(numIters):
    T = 0.5*(3.0*I - Z.bmm(Y))
    Y = Y.bmm(T)
    Z = T.bmm(Z)
  sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
  error = compute_error(A, sA)
  return sA, error

# Backward via iterative Lyapunov solver
def lyap_newton_schulz(z, dldz, numIters, dtype):
  batchSize = z.shape[0]
  dim = z.shape[1]
  normz = z.mul(z).sum(dim=1).sum(dim=1).sqrt()
  a = z.div(normz.view(batchSize, 1, 1).expand_as(z))
  I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
  q = dldz.div(normz.view(batchSize, 1, 1).expand_as(z))
  for i in range(numIters):
    q = 0.5*(q.bmm(3.0*I - a.bmm(a)) - a.transpose(1, 2).bmm(a.transpose(1,2).bmm(q) - q.bmm(a)) )
    a = 0.5*a.bmm(3.0*I - a.bmm(a))
  dlda = 0.5*q
  return dlda           

# Create random PSD matrix
def create_symm_matrix(batchSize, dim, numPts, tau, dtype):
    A = torch.zeros(batchSize, dim, dim).type(dtype)
    for i in range(batchSize):
        pts = np.random.randn(numPts, dim).astype(np.float32)
        sA = np.dot(pts.T, pts)/numPts + tau*np.eye(dim).astype(np.float32);
        A[i,:,:] = torch.from_numpy(sA);
    print('Creating batch %d, dim %d, pts %d, tau %f, dtype %s' %(batchSize, dim, numPts, tau, dtype))
    return A

# Command line arguments
parser = argparse.ArgumentParser(description='Matrix squareroot and its gradient demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--pts', type=int, default=1000, metavar='N',
                    help='number of points to construct covariance matrix (default: 1000)')
parser.add_argument('--tau', type=float, default=1.0, metavar='N',
                    help='conditioning by adding to the diagonal (default: 1.0)')
parser.add_argument('--num-iters', type=int, default=10, metavar='N',
                    help='number of schulz iterations (default: 5)')
parser.add_argument('--dim', type=int, default=64, metavar='N',
                    help='size of the covariance matrix (default: 64)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
if args.cuda:
  d = torch.cuda.FloatTensor
else:
  d = torch.FloatTensor

# Create matrix and gradient randomly
A = Variable(create_symm_matrix(batchSize=args.batch_size, dim=args.dim, numPts=args.pts, tau=args.tau, dtype=d), requires_grad=True)
dldz = Variable(torch.randn(args.batch_size, args.dim, args.dim).type(d), requires_grad=False)
dldz = 0.5*(dldz + dldz.transpose(1,2));

# Forward + backward with SVD
# Time: O(n^3), Space: O(n^3)
print('Singular Value Decomposition (SVD):')
start=tm.time()
svd_sA, svd_grad, svd_error = sqrt_svd_lyap(A, -dldz, dtype=d)
end=tm.time()
svd_time = end-start;
print('  >> forward + backward time %fs, forward error %s' % (svd_time, svd_error.item()))


# Forward pass with Denman-Beavers iterations (no backward)
print('Denman-Beavers iterations (%i iters) ' % (args.num_iters))
start = tm.time()
sA, error = sqrt_denman_beavers(A, args.num_iters, dtype=d);
end = tm.time()
print('  >> forward time %fs, error %s' % (end-start, error.item()))
print('  >> no backward via autograd')

# Forward pass with Newton-Schulz (autograd version)
# Time: O(Tn^2), Space: O(Tn^2), with T iterations
print('Newton-Schulz iterations (%i iters) ' % (args.num_iters))
start = tm.time()
sA, error = sqrt_newton_schulz_autograd(A, args.num_iters, dtype=d);
end = tm.time()
iter_time = end-start;
print('  >> forward: time %fs, error %s' % (end-start, error.item()))

# Backward pass with autograd
start = tm.time()
#with torch.autograd.profiler.profile() as prof:
sA.backward(dldz)
#print(prof)
end = tm.time()
iter_time += end-start
backward_error = svd_grad.dist(A.grad.data)
print('  >> backward via autograd: time %fs, error %f' % (end-start, backward_error))
print('  >> speedup over SVD: %.1fx' %(svd_time / iter_time))

# Forward pass with Newton-Schulz
# Time: O(Tn^2), Space: O(n^2), with T iterations
print('Newton-Schulz iterations (foward + backward) (%i iters) ' % (args.num_iters))
start = tm.time()
sA, error = sqrt_newton_schulz(A.data, args.num_iters, dtype=d);
end = tm.time()
iter_time = end-start
print('  >> forward: time %fs, error %s' % (end-start, error))

# Backward pass with Newton-Schulz
start = tm.time()
dlda = lyap_newton_schulz(sA, dldz.data, args.num_iters, dtype=d)
end = tm.time()
iter_time += end-start
backward_error = svd_grad.dist(dlda)
print('  >> backward: time %fs, error %f ' % (end-start, backward_error))
print('  >> speedup over SVD: %.1fx' %(svd_time / iter_time))
