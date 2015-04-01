---
layout:     post
title:      "Valuing American Options by the Longstaff-Schwartz Approach"
subtitle:   "A Simple Least-Squares Approach"
date:       2015-03-14 12:00:00
author:     "Aetienne Sardon"
header-img: ""
---
In the well known paper "Valuing American Options by Simulation: A Simple Least-Squares Approach" [1], Francis
Longstaff and Eduardo Schwartz present an intuitive and straightforward approach how to value American options.
They thereby follow a dynamic programming approach, such that they - based on simulation results - first consider the options final payoffs
and then recursively determine the expected continuation value by regressing the next period's payoff on the previous one. 
Based on the expected continuation value one can then decide whether to early exercise or not.

The steps of the valuation algorithm are as follows:
- model stock prices (e.g. by Brownian motion)
- determine the options final payoffs
- iterate backwards through the simulated stock price path
- determine in-the-money (ITM) option values of current period
- regress current ITM option value on discounted future option value (e.g. by quadratic regression $Y = a + bX + cX^2$
- if current ITM option value (early exercise value) is greater than the value from continuation (fitted value from regression), then do early exercise
- based on the recursion, determine optimal exercise times
- based on exercise times, determine expected payoff and take the average as the final option price

Below I have implemented the Longstaff-Schwartz approach in Python. At the bottom you can
also see that I have used the same example as proposed in their paper. When running the module
one can see that the value of the American option is 0.114434330045, just as in the paper.

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt

def run_brownian_motion_mc(init_price, risk_free_rate, sigma, T, K, dt=1.0):
    _tmp_rand = np.append(np.zeros((K,1)), np.random.normal(risk_free_rate, sigma, K*(T-1)).reshape((K,T-1)), axis=1)
    _prices = init_price * np.cumprod(np.exp(_tmp_rand * dt), axis=0)
    return _prices
    
def longstaff_schwartz(mc_prices, strike_price, is_call=True, risk_free_rate=0.03, dt=1.0):
    K, T = mc_prices.shape
    _exercise = np.zeros(mc_prices.shape)
    _payoff = np.zeros(mc_prices.shape)
    
    if(is_call):
        _in_the_money = (mc_prices - strike_price) > 0
        _payoff[_in_the_money] = mc_prices[_in_the_money] - strike_price
    else:
        _in_the_money = (strike_price - mc_prices) > 0
        _payoff[_in_the_money] = strike_price - mc_prices[_in_the_money]

    #---- recursion ----#
    for t in range(T-2,-1,-1):
        if(np.sum(_in_the_money[:,t]==1)>0):
            #--- get early exercise payoff ---#
            _x = _payoff[_in_the_money[:,t], t]
            
            #--- get next perios continuation value ---#
            _y = _payoff[_in_the_money[:,t], t+1] * np.exp(-risk_free_rate * dt)
            
            #--- regress early exercise on continuation value ---#
            _coef = np.polynomial.polynomial.polyfit(_x, _y, 2) # conditional expectatio
            _cond_exp = _coef[0] + _coef[1] * _x + _coef[2] * _x ** 2
            
            #--- compare early exercise with continuatio ---#
            _exercise[_in_the_money[:,t],t] = _cond_exp < _payoff[_in_the_money[:,t],t]
            _payoff[_in_the_money[:,t],t] = _payoff[_in_the_money[:,t],t] * _exercise[_in_the_money[:,t],t] 

    _exercise[np.cumsum(_exercise, axis=1) > 1] = 0 # set subsequent exercise dates equal to zero
    _exercise[np.sum(_exercise, axis=1) == 0, -1] = 1 # if no early exercise, then exercise at maturity

    _discount_factor = np.exp(np.arange(0,T) * -risk_free_rate * dt)
    _exp_val = np.sum(_exercise * _payoff * _discount_factor) / K
    
    return _exp_val
    
#test1 = run_brownian_motion_mc(100, 0.001, 0.01, 100, 1000)
test = np.array([
    [1, 1.09, 1.08, 1.34],
    [1, 1.16, 1.26, 1.54],
    [1, 1.22, 1.07, 1.03],
    [1, 0.93, 0.97, 0.92],
    [1, 1.11, 1.56, 1.52],
    [1, 0.76, 0.77, 0.90],
    [1, 0.92, 0.84, 1.01],
    [1, 0.88, 1.22, 1.34],
])
plt.plot(test)
print longstaff_schwartz(test, 1.1, False, 0.06)
{% endhighlight %}

## References
[1] Longstaff, F.R. and Schwartz, E.S. (2001): "Valuing American Options by Simulation: A Simple Least-Squares Approach", The Review of Financial Studies, Vol. 14, No. 1, pp. 113-147