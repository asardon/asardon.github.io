---
layout:     post
title:      "Risk Neutral Valuation"
subtitle:   "Reviewing the Basics"
date:       2015-01-12 12:00:00
author:     "Aetienne Sardon"
header-img: "img/home-bg.jpg"
---
<p>When I first encountered the concept of "risk neutral valuation" I found it kind of abstract and hard to wrap my head around. The main problem I think is that we are used to look at the world in terms of real (in finance we call these "physical") probabilities. Whenever we face an investment decision we naturally consider the expected payoff under the real (physical) probability distribution and discount the future expected cash flows by an appropriate discount rate that reflects the risks associated with the underlying investment. In mathematical terms we typically end up with something like the following:</p>

$$ V_0 = \sum_{t=1}^T \frac{ \mathbb{E}^{\mathbb{P}}[CF_t] }{ (1+r)^t } $$

<p>which is simply the standard discounted cash flow approach. So why don't we just follow the same methodology to value derivative contracts like a plain vanilla call option? Well of course we could determine the price of a call by</p>

$$ C_0 = \frac{1}{(1+r)^t} \mathbb{E}^{\mathbb{P}}[(S_t-K)^+]  $$

<p>but here we face the problem that we must find an adequate discount rate $r=r_f+\pi$ that incorporates the risk of the call option. Note that market participants might differ regarding the risk premium $\pi$ they demand for buying a call option. Although I'm not saying it's impossible to determine individual risk premia it turns out that there is a way how we can price an asset without having to know anything about the risk preferences of market participants and that's (yes, you've guessed it) by means of risk neutral valuation.</p>

<p>Bear in mind that the value of a derivative is fully determined or derived (as the name suggests) from the value of the underlying asset, in this case $S_t$. So what do we know about $S_t$? We know that today's price $S_0$ should be equal to the (real) expected future value $\mathbb{E}^{\mathbb{P}}[S_t]$ discounted at the risk free rate plus a risk premium $r=r_f+\pi$. However, we can also alter the probabilities attributed to each future outcome in such a way that the current price $S_0$ is equal to this new expectation $\mathbb{E}^{\mathbb{P}}[S_t]$ discounted solely at the risk free rate $r_f$ without any risk premium. This leads to:</p>
$$ S_0 \overset{!}{=} \frac{ \mathbb{E}^{\mathbb{P}}[S_t] }{1+r_f+\pi} \Leftrightarrow S_0 \overset{!}{=} \frac{ \mathbb{E}^{\mathbb{Q}}[S_t] }{1+r_f} $$
<p>i.e. we introduce a new probability measure $\mathbb{Q}$ under which our expected future stock price equals the amount we get if we simply invest $S_0$ at the risk free rate $r_f$. We have constructed $\mathbb{Q}$ in such a way that the risk premium is fully incorporated into the expectation and is eliminated from the discount factor, thus $\mathbb{Q}$ is called the risk neutral measure. We can reconstruct the value of the current stock price by:</p>
$$ S_0 = \frac{ \mathbb{E}^{\mathbb{Q}}[S_t] }{ (1+r)^t } =  \frac{ S_0 (1+r_f)^t }{(1+r)^t} = S_0 $$
<p>Note that $\mathbb{P}$ must be an equivalent probability measure to $\mathbb{Q}$, but that's just a fancy term for saying that events that have zero probability $\mathbb{P}$ must also have zero probability under $\mathbb{Q}$ and vice versa. </p>

<p>Naturally, we think of probabilities as a tool for predicting future outcomes. However, it's misleading to think of the risk neutral probabilities in this sense. 
Instead, $\mathbb{E}^{\mathbb{Q}}[\cdot]$ can be seen as a special weighting function where we simply assign specific weights $w_i$ to each possible outcome value $x_i$, i.e.</p>
$$\mathbb{E}^{\mathbb{Q}}[X]=\sum_{i=0}^N x_i w_i$$
<p>(or in the continuous case $\mathbb{E}^{\mathbb{Q}}[X]=\int_{-\infty}^{+\infty} x w(x) dx$) where all $w_i$ happen to be $0\leq w_i \leq1$ and $\sum w_i=1$. Thus we can refer to the weights as probabilities (they clearly exhibit properties of the like), however they don't match our (or at least my) everyday intuition of probabilities. Instead I would argue that they are rather the result when imposing the condition $S_0 \overset{!}{=} \frac{ \mathbb{E}^{\mathbb{Q}}[S_t] }{1+r_f}$. </p>

<p>The great thing about risk neutral pricing is that once we have constructed the risk neutral measure $\mathbb{Q}$ we can simply value derivatives by using the artificial probability distribution of $\mathbb{Q}$. Thus, the risk neutral valuation framework provides us a tool to price derivatives without having to know anything about the true probability distribution $\mathbb{P}$ of the underlying asset. And by applying the risk neutral valuation we can be sure that the derivative prices we obtain from $\mathbb{Q}$ are consistent with the current prices of the underlying (i.e. no arbitrage opportunities exist).</p>

<p>So let's consider the standard textbook example for valuing a call option in a one period binary world scenario. Consider a stock currently trading at $S_0=100$ which in the next period $t=1$ can either move up to $S_1^U=120$ or down to $S_1^D=80$. Let the risk free rate be at $r_f=0.05$. Let us first derive $\mathbb{Q}$, i.e. determine the probabilities that fulfill:</p>
$$ \mathbb{E}^{\mathbb{Q}}[S_1] = S_1^U p + S_1^D(1-p) \overset{!}{=} S_0 (1+r_f) $$
<p>Simply rearranging gives us:</p>
$$p=\frac{ S_0 (1+r_f) - S_1^D }{S_1^U-S_1^D}=\frac{105-80}{120-80}=0.625$$
<p>Finally, we use the probability distribution under $\mathbb{Q}$ to derive a call with strike $K=100$:</p>
$$C_0=\frac{ \mathbb{E}^{\mathbb{Q}}[C_1] }{1+r_f}=\frac{1}{1+r_f}( (S_1^U-K)^+ p + (S_1^D-K)^+(1-p))=\frac{ 20 \cdot 0.625 }{1.05}=11.91$$
<p>So we see that the value of the call is 11.91 and no information regarding $\mathbb{P}$ was required to arrive at this price. Note that this price of course matches the value we obtain by replicating the future call values by a portfolio consisting of stocks and bonds, i.e.</p>

$$
 \begin{pmatrix}
  120 & 105  \\
  80 & 105 \\
 \end{pmatrix}
 \begin{pmatrix}
x_1 \\
x_2
 \end{pmatrix}
\overset{!}{=}
 \begin{pmatrix}
20 \\
0
 \end{pmatrix}
 $$

<p>where after inverting the future asset price matrix we obtain for the asset allocation vector $\vec{x}=(0.5,-0.3809)^T$. Thus, the price indicated by the replicating approach equals our former result $(S_0, B_0)^T \cdot \vec{x}=11.91$.</p>

<p>Any deviation from $C_0$ would allow for arbitrage opportunities. For example, if the call would be trading at a discount, let's say $C_0=10.0$, then we could simply sell the stock/bond portfolio with allocations $\vec{x}$ for 11.91 in the market (i.e. sell $0.5$ in stocks and buy $0.3809$ in bonds) and buy the call for $C_0=10.0$. Thus, regardless of the future outcome we would net in a riskless profit of $11.91-10.0=1.91$.</p>

<p>We see that the risk neutral valuation approach is just one of many ways how we can price derivatives. Not surprisingly, in the risk neutral valuation framework we obtain the same prices as we do when following a replication approach (luckily no ambiguity in prices!). So, that's it for today! I hope you found the post helpful!</p>