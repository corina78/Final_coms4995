# Final_coms4995
README

The current branch of the project is called experiments_timeit and contains the code that perfoms davidon optimization algorithm for Minst dataset. 

The main script from where to launch the code is davidon_update.py. It requires two updates in order to run. In line 210 define the path to the files. In line 235 define what are the units in layer that will be required. All the tests in the project have been performed with two architectures:
Model a : [784,64,32,10]
Model b : [784,128,64,10]


The objective of this exercise is to evaluate and compare the performance of Stochastic Gradient Descent (SGD) and Davidson's method, not focusing specifically on metrics such as accuracy. Due to the extensive time required for training, the number of iterations has been limited to 100. This limitation means that the accuracy achieved is not optimal, approximately around 40%. However, this exercise still serves as a benchmark for comparing the efficiency and effectiveness in cost reduction of the two algorithms. The results suggest that the rate of cost reduction with Davidson's method is twice as fast as that achieved with Stochastic Gradient Descent (SGD), implying a more rapid decrease in cost, but the algorithm exits before reaching a global minimum.

Other branches in the repository: master branch runs the baseline SGD solution, experiments_sparse runs some experiments using sparse matrices.
