## Matrix square root and its gradient


### Overview

This repository contains Python and Matlab code for computing the matrix square root (ZZ = A) and its gradient using various techniques on the GPU. For the forward computation (the square root of a matrix), SVD and iterative methods are implemented. For the backward computation (the gradient of the square root) matrix backprop, Lyapunov solver using SVD, autograd via the forward iterative method, and Lyapunov solver using iterative methods are implemented. 

These implementations offer different numerical accuracy, memory overhead, and speed tradeoffs. See the output of the  python code below as an example. In general the batch-mode versions of the iterative forward and backward methods are an order of magnitude faster than SVD-based methods.
We used these methods in our "improved bilinear pooling" paper. Please cite this if you use this code:

	@inproceedings{lin17improved,
		author = {Tsung-Yu Lin, and Subhransu Maji},
		booktitle = {British Machine Vision Conference (BMVC)},
		title = {Improved Bilinear Pooling with CNNs},
		year = {2017}}

Take a look at the BMVC paper and the main git repo for details on how these functions are used to improve the classification accuracy on various fine-grained recognition benchmarks. 

* Bilinear CNN project page: [http://vis-www.cs.umass.edu/bcnn](http://vis-www.cs.umass.edu/bcnn)
* Main git repository: [https://bitbucket.org/tsungyu/bcnn.git](https://bitbucket.org/tsungyu/bcnn.git)

#### Python code
The Python code is based on PyTorch. Supply --cuda option to run on a GPU. The following is the output on a Tesla K40 GPU.

**$ python matrix_sqrt.py --cuda** 

	Creating batch 128, dim 64, pts 1000, tau 1.000000, dtype <class 'torch.cuda.FloatTensor'>
	Singular Value Decomposition (SVD):
	  >> forward + backward time 1.401933s, forward error 1.60115939707e-06
	Denman-Beavers iterations (10 iters) 
	  >> forward time 8.109320s, error 7.49908053876e-07
	  >> no backward via autograd
	Newton-Schulz iterations (10 iters) 
	  >> forward: time 0.013402s, error 5.73769398216e-07
	  >> backward via autograd: time 0.017659s, error 0.000217
	  >> speedup over SVD: 45.1x
	Newton-Schulz iterations (foward + backward) (10 iters) 
	  >> forward: time 0.012401s, error 5.73769398216e-07
	  >> backward: time 0.004554s, error 0.000238 
	  >> speedup over SVD: 82.7x


The speedup over the SVD-based method is smaller on the CPU as seen below:

**$ python matrix_sqrt.py**

	Creating batch 128, dim 64, pts 1000, tau 1.000000, dtype <class 'torch.FloatTensor'>
	Singular Value Decomposition (SVD):
	  >> forward + backward time 1.356299s, forward error 2.23704591917e-06
	Denman-Beavers iterations (10 iters) 
	  >> forward time 0.408686s, error 7.41325095532e-07
	  >> no backward via autograd
	Newton-Schulz iterations (10 iters) 
	  >> forward: time 0.131901s, error 5.82883330935e-07
	  >> backward via autograd: time 0.219422s, error 0.000294
	  >> speedup over SVD: 3.9x
	Newton-Schulz iterations (foward + backward) (10 iters) 
	  >> forward: time 0.127630s, error 5.82883321609e-07
	  >> backward: time 0.225944s, error 0.000309 
	  >> speedup over SVD: 3.8x

#### Matlab code

The Matlab code does not support batch-mode matrix multiplication hence the speedups are smaller over the PyTorch version. It also does not include the autograd version for computing the gradients.

**$ matrix_sqrt**

	Experiment:
	>>>> n 512, tau 1.0, numPoints 1000, float 1, gpu 1, maxIter 10, runIter 10
	>>>> floating point: single
	>>>> GPU: true
	
	Forward time:
	 0.509482s SVD
	 1.619821s Denman-Beavers (10 iter)
	 0.135493s Newton-Schulz (10 iter)

	Backward time:
	 0.048150s Matrix-backprop
	 0.021147s Lyapunov SVD
	 0.216029s Lyapunov Newton-Schulz (10 iter)

	Forward error:
	 1.767662e-06 SVD
	 3.807475e-07 Denman-Beavers (10 iter)
	 1.829006e-06 Newton-Schulz (10 iter)

	Backward error:
	 1.731067e-03 Matrix-backprop
	 4.986763e-06 Lyapunov SVD
	 1.484792e-02 Lyapunov Newton-Schulz (10 iter)

