---
layout:     post
title:      "Baum Welch Algorithm"
subtitle:   "Part 2"
date:       2015-01-23 12:00:00
author:     "Aetienne Sardon"
header-img: "img/home-bg.jpg"
---
<p>In the last post we went through the derivation of the Baum-Welch algorithm, 
which allows us to estimate/learn the parameters of a Hidden Markov Model (HMM). 
In this post I would like to illustrate how we can translate our HMM into Python code. So let's dive right into it!</p>

<p>First, let's import some packages we will need later on and define a HMM class with the following attributes: number of states, number of mixture components, initial state distribution, transition matrix and Gaussian mixture parameters.</p>

	from scipy.stats import norm
	import numpy as np
	import matplotlib.pyplot as plt
	
	class HMM():
	    
	    def __init__(self, num_states, num_components):
	        self.num_states = num_states
	        self.num_components = num_components
	        self.init_prob = None
	        self.trans_prob = None
	        self.means = None
	        self.variances = None
	        self.mixture_weights = None
	
	    def set_init_prob(self, init_prob):
	        self.init_prob = init_prob
	        
	    def set_trans_prob(self, trans_prob):
	        self.trans_prob = trans_prob
	        
	    def set_means(self, means):
	        self.means = means
	        
	    def set_variances(self, variances):
	        self.variances = variances
	        
	    def set_weights(self, weights):
	        self.mixture_weights = weights
	        self.m = len(weights)

<p>Pretty much self-explanatory code. Next, let's also implement two functions which allow us to calculate the Gaussian mixture emission probabilities.</p>

	class HMM():
		...  
	    def single_emission_prob(self, obs):
	        out = np.zeros((len(obs), self.num_states, self.num_components))
	
	        for i in range(self.num_states):
	            for l in range(self.num_components):
	                out[:,i,l] = norm.pdf(obs, self.means[i][l], self.variances[i][l]) + np.random.random_sample() * 1.0e-270
	        return out
	       
	    def mixture_emission_prob(self, single_emission_prob):
	        out = np.zeros((len(single_emission_prob), self.num_states))
	        
	        for i in range(self.num_states):
	            out[:,i] = np.sum( np.multiply(self.mixture_weights[i], single_emission_prob[:,i,:]), axis=1)
	        return out

<p>The reason why we split the calculation of the Gaussian mixtures into two functions is because we will also need the single Gaussian probabilities later on, when determining the mixture parameters. Note that we add some noise to the single gaussian emissions in order to avoid the occurrence of 0% probabilities (which would lead to difficulties when computing $\gamma_{i,l}(t)$ as we here need to divide by $\sum_{l=1}^m b_{i,l}(o_t)$.)</p>

<p>Ok, now that we've defined the basis of our HMM class, let's turn to the actual heart of the Baum-Welch algorithm, i.e. updating our model parameters. The update equations we are going to use are as follows:</p>

$$
\begin{eqnarray} 
\pi_{i} &=& \gamma_i(1) \\
a_{i,j} &=& \frac{  \sum_{t=1}^{T-1} \alpha_i(t)  a_{i,j} b_{j}(o_{t+1}) \beta_j(t+1)  }{ \sum_{t=1}^{T-1} \sum_{j=1}^n \alpha_i(t)  a_{i,j} b_{j}(o_{t+1}) \beta_j(t+1) } \\
w_{i,l} &=&  \frac{ \sum_{t=1}^{T} \gamma_{i,l}(t) }{ \sum_{t=1}^{T} \gamma_i(t) } \\
\mu_{i,l} &=&  \frac{ \sum_{t=1}^{T} \gamma_{i,l}(t) o_t }{ \sum_{t=1}^{T} \gamma_{i,l}(t) } \\
\sigma_{i,l}^2 &=&  \frac{ \sum_{t=1}^{T} \gamma_{i,l}(t) (o_t - \mu_{i,l} ) ^2 }{ \sum_{t=1}^{T} \gamma_{i,l}(t) } \\
\end{eqnarray}
$$

<p>Where $\gamma_i(t)$ and $\gamma_{i,l}(t)$ are defined as:</p>

$$
\begin{eqnarray} 
\gamma_i(t) &:=& \frac{ \alpha_i(t) \cdot \beta_i(t) }{ \sum_{i=1}^n \alpha_i(t) \cdot \beta_i(t) }  \\
\gamma_{i,l}(t) &:=& \gamma_i(t) \frac{ w_{i,l} b_{i,l}(o_t) }{ \sum_{l=1}^m b_{i,l}(o_t)  }  \\
\end{eqnarray} 
$$

<p>And $\alpha_i(t)$ and $\beta_i(t)$ are defined as:</p>
$$
\begin{eqnarray} 
\alpha_i(t) &=& b_{i}(o_t) \sum_{j=1}^{|\mathcal{S}|} \alpha_j(t-1) \cdot a_{j,i}  \quad,  \quad \alpha_i(1) = \pi_i b_i(o_1) \\
\beta_i(t) &=& \sum_{j=1}^{|\mathcal{S}|} a_{i,j} \cdot b_{j}(o_{t+1}) \cdot \beta_j(t+1) \quad, \quad \beta_i(T)=1\\
\end{eqnarray} 
$$

<p>We see that the forward and backward probabilities $\alpha_i(t)$ and $\beta_i(t)$ basically show up in every equation, so let's start defining functions for these.</p>

## Forward/Backward Probabilities
<p>Before we can continue, there's an important remark to be made. When calculating the forward and backward probabilities we will sooner or later encounter the problem of underflow. Underflow occurs when we operate with numbers that are too small to be represented by our computer. When calculating the forward/backward probabilities we recursively multiply numbers that are smaller than one. If we do that often enough the resulting values will become very small pretty fast, e.g. for a non trivial dataset with a sequence length of $T=1000$ we might be working with numbers in the order of $\Pr(\cdot)^{1000} \propto 10^{-1000}$. Numpy supports double precision floats (float64), which have 1 sign bit, 11 exponent bits, 52 mantissa bits. So the exponent bit can store $2^{11}-2=2048-2=2046$ different values (subtracting 2 because the smallest and largest exponent are used to represent signed zero and infinity) such that the representable values are in the range of $[2^{-1022}, 2^{+1023}]\approx[10^{-307},10^{+308}]$ (when using subnormal numbers actually $[10^{-324},10^{+308}]$). Obviously, in our example $10^{-1000}$ will cause an underflow. But luckily, there is a way how we can solve this problem. One way is to rescale the forward/backward probabilities, i.e.:</p>

$$
\begin{eqnarray} 
\hat{\alpha}_i(t) &=& \frac{\alpha_i(t)}{\sum_{i=1}^N \alpha_i(t)} \\
\hat{\beta}_i(t) &=& \frac{\beta_i(t)}{\sum_{i=1}^N \beta_i(t)} \\
\end{eqnarray} 
$$

It can be shown that the update equations remain unchanged when using the rescaled forward/backward probabilities $\hat{\alpha}_i(t), \hat{\beta}_i(t)$ instead of $\alpha_i(t), \beta_i(t)$ [[1]](#References).

<p>So now that we have tackled the underflow problem, let's write two functions for the modified/rescaled forward and backward probabilities. Given a sequence of gaussian mixture probabilities we want to calculate $\hat{\alpha}_i(t)$ and $\hat{\beta}_i(t)$:</p>

	class HMM():
		...
	    def alpha(self, mixture_emission):
	        alpha = np.zeros((len(mixture_emission), self.num_states))
	        alpha[0,:] = np.multiply( self.init_prob, mixture_emission[0,:] )
	        
	        c = np.zeros((len(mixture_emission)))
	        c[0] = 1/np.sum(alpha[0])
	        alpha[0,:] = alpha[0]*c[0]
	        
	        for t in range(1, len(mixture_emission)):
	            for i in range(self.num_states):
	                alpha[t,i] = mixture_emission[t,i] * np.inner( alpha[t-1,:], self.trans_prob[:,i] )
	                alpha[t,i] = 0 if(np.isnan(alpha[t,i])) else alpha[t,i]
	            c[t] = 1/np.sum(alpha[t])
	            alpha[t] = alpha[t]*c[t]
	        return alpha, c
	    
	    def beta(self, mixture_emission):
	        beta = np.zeros((len(mixture_emission), self.num_states))
	        beta[-1,:] = 1
	        
	        c = np.zeros((len(mixture_emission)))
	        c[-1] = 1/np.sum(beta[-1])
	        beta[-1,:] = beta[-1,:]*c[-1]
	
	        for t in range(len(mixture_emission)-2,-1,-1):
	            for i in range(self.num_states):
	                beta[t,i] = np.sum( np.multiply(mixture_emission[t,i], np.multiply( self.trans_prob[i,:], beta[t+1,:] ) ) )
	                beta[t,i] = 0 if(np.isnan(beta[t,i])) else beta[t,i]
	            c[t] = 1/np.sum(beta[t])
	            beta[t] = beta[t]*c[t]
	        return beta, c

<p>Basically, we first initialize an empty numpy array and then successively fill in the forward/backward probabilities by iterating over the observations and different states.</p> 

## Helper Variables
<p>Now that we've defined functions for $\hat{\alpha}_i(t)$ and $\hat{\beta}_i(t)$ we can use these values as an input for calculating $\gamma_i(t)$:</p>

	class HMM():
		...        
	    def gamma(self, alpha_scaled, beta_beta):
	        gamma = np.zeros((len(alpha_scaled), self.num_states))
	        
	        for i in range(self.num_states):
	            gamma[:,i] = alpha_scaled[:,i] * beta_beta[:,i]
	            gamma[:,i] = np.nan_to_num(gamma[:,i])
	        gamma[np.sum(gamma == 0, axis=1)>0] += 0.1
	        gamma = gamma / np.sum(gamma, axis=1).reshape((len(gamma),1))
	        
	        return gamma

Here we iterate through the number of states and use the vectorized numpy multiplication to determine $\gamma_1(t), ..., \gamma_n(t)$. Note that we apply additive smoothing (or pseudocounts) in order to avoid zero-frequency problems. Given $\gamma\_i(t), b_{i,l}(o\_t)$ and $b\_{i}(o\_t)$ we can now also calculate $\gamma\_{i,l}(t)$:
 
	class HMM():
		...      
	    def gamma_w(self, gamma, single_emission, mixture_emission):
	        gamma_w = np.zeros((len(gamma), self.num_states, self.num_components))
	        
	        for i in range(self.num_states):
	            for l in range(self.num_components):
	                gamma_w[:,i,l] = single_emission[:,i,l] * self.mixture_weights[i,l]
	                gamma_w[:,i,l] = np.nan_to_num(gamma_w[:,i,l])
	            gamma_w[np.sum(gamma_w[:,i,:] == 0, axis=1)>0,i] += 0.1
	
	            gamma_w[:,i] = gamma[:,i,None] * gamma_w[:,i] / np.sum(gamma_w[:,i], axis=1)[:,None]
	
	        return gamma_w

Here, we again use additive smoothing.

## Updating Model Parameters
Based on $\hat{\alpha}\_i(t), \hat{\beta}\_i(t), \gamma\_i(t), \gamma\_{i,l}(t), b\_{i,l}(o\_t)$ and $b\_{i}(o\_t)$ we can now finally determine the parameter updates for $\pi\_{i}, a\_{i,j}, w\_{i,l}, \mu\_{i,l}$ and $\sigma\_{i,l}^2$.

	class HMM():
		... 
	    def update_init_prob(self, alpha_scaled, beta_scaled):
	        numerator = alpha_scaled[0,:] * beta_scaled[0,:]
	        denominator = np.sum(numerator)
	        self.init_prob = numerator / denominator
	          
	    def update_trans_prob(self, alpha_scaled, beta_scaled, mixture_emission):
	        for i in range(self.num_states):
	            for j in range(self.num_states):
	                self.trans_prob[i,j] = np.sum( alpha_scaled[:-1,i] * self.trans_prob[i,j] * mixture_emission[1:,j] * beta_scaled[1:,j] )
	
	            self.trans_prob[i,:] = self.trans_prob[i,:] / np.sum( self.trans_prob[i,:] )
	
	    def update_weights(self, gamma, gamma_w):
	        for i in range(self.num_states):
	            self.mixture_weights[i,:] = np.sum(gamma_w[:,i,:], axis=0) / np.sum( gamma[:,i], axis=0)
	           
	    def update_means(self, gamma_w, obs):
	        for i in range(self.num_states):
	            for l in range(self.num_components):
	                self.means[i,l] = np.sum(gamma_w[:,i,l] * obs, axis=0) / np.sum( gamma_w[:,i,l], axis=0)
	   
	    def update_variances(self, gamma_w, means, obs):
	        var = np.var(obs)
	        min_var = var * 0.001
	        max_var = var 
	        for i in range(self.num_states):
	            for l in range(self.num_components):
	                self.variances[i,l] = np.sum(gamma_w[:,i,l] * ((obs - means[i,l])**2), axis=0) / np.sum( gamma_w[:,i,l], axis=0)
	                if(np.isnan(self.variances[i,l])):
	                    self.variances[i,l] = min_var
	                if(self.variances[i,l]<min_var):
	                    self.variances[i,l] = min_var
	                if(self.variances[i,l]>max_var):
	                    self.variances[i,l] = max_var

Here we pretty much simply implement the previously defined parameter update equations. For the variance we also handle cases where the variance might be zero due to zero-frequency problems.  

## Toy Example
Let's create some dummy data to test our Python HMM. Therefore let's define the true model parameters $\pi\_{i}, a\_{i,j}, w\_{i,l}, \mu\_{i,l}$ and $\sigma\_{i,l}^2$ from which we want to sample.

	np.random.seed(2)
	
	trans_prob = np.array([
	    [0.9, 0.1],
	    [0.8, 0.2]
	    ])
	weights = np.array([
	    [0.7, 0.3],
	    [0.3, 0.7]
	    ])
	means = np.array([
	    [0.05, 0.1],
	    [-0.05, -0.15]
	    ])
	variances = np.array([
	    [0.05, 0.02],
	    [0.04, 0.02]
	    ])
	
	rand1 = np.random.random_sample((1000))
	rand2 = np.random.random_sample((1000))
	
	s = np.zeros((len(rand1)))
	o = np.zeros((len(rand1)))
	
	s[0] = 1
	
	for t in range(1, len(rand1)):
	    s[t] = np.random.choice(2, 1, p=trans_prob[s[t-1],])[0]
	    idx = np.random.choice(2, 1, p=weights[s[t]])[0]
	    o[t] = norm.rvs(means[s[t], idx], variances[s[t], idx], 1)
	
	x = np.arange(0, len(o))
	plt.plot(x[s==0], o[s==0], 'bo')
	plt.plot(x[s==1], o[s==1], 'ro')

The data should look like this: ![Alt]({{ site.baseurl}}/img/2015-01-27-Baum-Welch-Algorithm-Part-2_dummy data.png)

Now let's make some prior guess of the model parameters and check how well the Baum-Welch is able to infer the true parameter values.

	init_prob = np.array([0.5, 0.5])
	trans_prob = np.array([
	    [0.5, 0.5],
	    [0.5, 0.5]
	    ])
	weights = np.array([
	    [0.5, 0.5],
	    [0.5, 0.5]
	    ])
	means = np.array([
	    [0.1, 0.01],
	    [-0.1, -0.1]
	    ])
	variances = np.array([
	    [1.0, 1.0],
	    [1.0, 1.0]
	    ])
	
	test = HMM(2, 2)
	test.set_means(means)
	test.set_variances(variances)
	test.set_weights(weights)
	test.set_init_prob(init_prob)
	test.set_trans_prob(trans_prob)
	
	obs = o 
	
	for k in range(0,100):
	    pre_trans_prob = np.copy(test.trans_prob)
	    pre_means = np.copy(test.means)
	    pre_variances = np.copy(test.variances)
	
	    single_emission_prob = test.single_emission_prob(obs)
	    mixture_emission_prob = test.mixture_emission_prob(single_emission_prob)
	    alpha_scaled, c = test.alpha(mixture_emission_prob)
	    beta_scaled, d = test.beta(mixture_emission_prob)
	    gamma = test.gamma(alpha_scaled, beta_scaled) 
	    gamma_w = test.gamma_w(gamma, single_emission_prob, mixture_emission_prob) 
	    means = test.means
	
	    test.update_init_prob(alpha_scaled, beta_scaled)
	    test.update_trans_prob(alpha_scaled, beta_scaled, mixture_emission_prob)
	    test.update_weights(gamma, gamma_w)
	    test.update_means(gamma_w, obs)
	    test.update_variances(gamma_w, means, obs)
	
	    if(np.linalg.norm(pre_trans_prob-test.trans_prob)<0.00000000001):
	        print "Convergence Criterion fulfilled at iteration: %d" % k
	        break
	    
	    if(k%10==0):
	        print "Result"
	        print test.init_prob
	        print test.trans_prob        
	        print test.mixture_weights               
	        print test.means
	        print test.variances

 The output should look something like:

	Result
	[ 0.  1.]
	[[  9.99999997e-01   2.95075911e-09]
	 [  9.99999999e-01   1.16465859e-09]]
	[[ 0.93782548  0.06217452]
	 [ 0.9270645   0.0729355 ]]
	[[ 0.07487803  0.10608365]
	 [-0.05272813  0.07537741]]
	[[ 0.00194984  0.00101702]
	 [ 0.00591209  0.00063417]]
	Convergence Criterion fulfilled at iteration: 36


Well, we got the the initial state distribution right, however, the transition probabilities seem a little extrem. The mixture weights and variances seem to be a little bit off but then the mixture means seem quite in line with the true model parameters. Note that we may be able to improve our results by altering our prior guesses. All in all the results of the Baum-Welch algorithm suggest that we should run several independent training sessions in order to receive more reliable estimates.
  
## References
[1] Rabiner, L.R. (1989): "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition", Proceedings of the IEEE, Vol. 77, No. 2.