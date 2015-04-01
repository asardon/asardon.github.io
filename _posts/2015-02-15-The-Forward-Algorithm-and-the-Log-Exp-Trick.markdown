---
layout:     post
title:      "The Forward Algorithm and the Log-Exp Trick"
subtitle:   "How to avoid underflows"
date:       2015-02-15 12:00:00
author:     "Aetienne Sardon"
header-img: ""
---
When implementing the forward algorithm, a problem bound to arise is that of underflow. 
In an [earlier post](http://asardon.github.io/2015/01/23/Baum-Welch-Algorithm-Part-2/) we discussed
how we can avoid underflows by using a scaling factor as proposed in [1]. However,
in this post we will show an alternative solution to the underflow problem by utilizing the
so called log-exp trick. A paper by [2] nicely explains how the log-exp can
be used in the context of Hidden Markov Models (HMM). In this post we will review the section of
his paper in which he shows how to apply the log-exp trick to the forward algorithm. But before we
continue, let us first briefly discuss what we mean by the log-exp trick.

Assume we would like to calculate the following quantity:

\begin{equation}
\begin{split}
z = \log \sum_{i=1}^n \exp(x_i)
\end{split}
\end{equation}

For an example, if $x_1=-1000, x_2=-999$ and $x_3=-998$ then $z=\log (\exp(-1000) + \exp(-999) + \exp(-998) )$ will result in $z=\log (0 + 0 + 0 )=-\infty$. However, we can also use the following identity

\begin{equation}
\log \sum_{i=1}^n \exp(x_i) = a + \log \sum_{i=1}^n \exp(x_i - a)
\end{equation}

where $a=\max_i(x_i)$. This means that the summands are shifted by the most significant $x_i$. This in turn allows us to get a more accurate result than $z=-\infty$. Applying the above mentioned identity results in $z=-998 + \log ( \exp(-2) + \exp(-1) + \exp(0) )\approx-997.82$.

In the case of the forward algorithm, the following quantities need to be computed:

\begin{equation}
\begin{split}
\alpha(i,1) &=& \pi(i) b(i,o_1) \\
\alpha(i,t) &=& b(i,o_t) \sum_{j=1}^n \alpha(j,t-1) a(i,j)
\end{split}
\end{equation}

or in log-domain:
\begin{equation}
\begin{split}
\log \alpha(i,1) &=& \log ( \log\pi(i) \cdot \log b(i,o_1)) \\
\log  \alpha(i,t) &=& \log ( \log b(i,o_t) \cdot \log \sum_{j=1}^n \log \alpha(j,t-1) \cdot \log a(i,j) )
\end{split}
\end{equation}

[2] provides nice definitions of "extended log/exp" functions that we can implement in order to compute $\log \alpha(i,1)$. In Python, these can be written as:

{% syntax python %}
import math
import numpy as np
from scipy.stats import norm

def get_eexp(x):
	if(x=="LOGZERO"):
		return 0
	else:
		return math.exp(x)
		
def get_eln(x):
	if(x==0):
		return "LOGZERO"
	elif(x>0):
		return math.log(x)
	else:
		print("negative input error")

def get_eln_list(x_list):
	_out = [0.0] * len(x_list)
	
	for i in range(len(x_list)):
		_out[i] = get_eln(x_list[i])
	return _out
	
def get_eln_list2d(x_list2d):
	_out = [[0.0] * len(x_list2d[0])] * len(x_list2d)
	
	for i in range(len(x_list2d)):
		_out[i] = get_eln_list(x_list2d[i])
	return _out

def get_elnsum(eln_x, eln_y):
	if(eln_x=="LOGZERO" or eln_y=="LOGZERO"):
		if(eln_x=="LOGZERO"):
			return eln_y
		else:
			return eln_x
	else:
		if(eln_x > eln_y):
			return eln_x + get_eln(1 + math.exp(eln_y - eln_x))
		else:
			return eln_y + get_eln(1 + math.exp(eln_x - eln_y))

def get_elnprod(log_x, log_y):
	if(log_x=="LOGZERO" or log_y=="LOGZERO"):
		return "LOGZERO"
	else:
		return log_x + log_y
{% endsyntax %}

Note that we do not use NumPy in order to work with lists that are able to handle different data types, such that we can identify "LOGZERO" values as strings. Although this can also be accomplished in NumPy it is not as straightforward as using the built-in list datatype. 

We can then use these "extended log/exp" functions for the forward algorithm in the following manner:

~~~ python
def get_log_alpha(ln_pi, ln_b, ln_a):
	num_states, T = len(ln_b), len(ln_b[0])
	ln_alpha = [[0] * T] * num_states 

	for i in range(num_states):
		ln_alpha[i][0] = get_elnprod(ln_pi[i], ln_b[i][0])
		
	for t in range(1, T):
		for j in range(num_states):
			for i in range(num_states):
				ln_alpha[i][t] = get_elnsum(ln_alpha[i][t], get_elnprod(ln_alpha[i][t-1], ln_a[i][j]))
			ln_alpha[j][t] = get_elnprod(ln_alpha[i][t], ln_b[j][t])
	return ln_alpha
~~~

In order to test the log forward algorithm we also need a function to calculate the emission probabilities. For a single Gaussian emission source, we get:

~~~ python
def get_single_gaussian_emission_probs(obs, mu, sd):
	num_states, T = len(mu), len(obs)
	out = [[0.0] * T] * num_states 

	for t in range(T):
		for i in range(num_states):
			out[i][t] = norm.pdf(obs[t], mu[i], sd[i]) #+ 1.0e-200
	return out
~~~

And lastly for comparison, the usual forward algorithm:

~~~ python
def get_alpha(pi, b, a):
	num_states, T = len(b), len(b[0])
	alpha = [[0] * T] * num_states 
	
	for i in range(num_states):
		alpha[i][0] = pi[i] * b[i][0]
		
	for t in range(1, T):
		for j in range(num_states):
			for i in range(num_states):
				alpha[i][t] += alpha[i][t-1] * a[i][j]
			alpha[j][t] *= b[j][t]
	return alpha    
~~~
	   
Finally, testing the above functions with synthetic data can be done as follows:

~~~ python
o = np.random.normal(0.05, 0.02, 10000)
pi = [0.2, 0.8]
b = get_single_gaussian_emission_probs(o, [0.03, 0.05], [0.01, 0.02]) 
a = [
	[0.3, 0.7],
	[0.8, 0.2]
]

ln_pi = get_eln_list(pi)
ln_b = get_eln_list2d(b)
ln_a = get_eln_list2d(a)

alpha1 = get_log_alpha(ln_pi, ln_b, ln_a)
alpha2 = get_eln_list2d(get_alpha(pi, b, a))
~~~

When running this test, one can see that "alpha2" will quickly return $+\infty$ values (in my case after the 135th element) while "alpha1" yields more precise results throughout the entire sequence. Thus, the exp-log trick can be used to solve the problem of underflow and allows us to use reasonable forward probabilities for HMM parameter estimation. I hope you found the post helpful!

## References
[1] Rabiner, L.R. (1989): "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition", Proceedings of the IEEE, Vol. 77, No. 2.

[2] Mann, T.P. (2006): "Numerically stable hidden markov model implementation", World Wide Web. Available in: http://bozeman.genome.washington.edu/compbio/mbt599_2006/hmm_scaling_revised.pdf