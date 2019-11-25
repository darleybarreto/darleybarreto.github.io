---
layout: post
title: Variational Inference (1) - Intro & Mean-Field Approximation
tags: [bayes-series, vi]
---

This is the #2 post in my series of posts about approximate Bayesian Inference (BI) where I will be talking about Variational Inference. If you do not understand most of the things here, I would recommend reading my previous post and/or the references. 
{: .text-justify}

Here is the outline of the blog posts of this series on BI:
1. [Introduction](/2019-09-22-intro-bayes).
2. Variational Inference (1) - Intro & Mean-Field Approximation (you are here).
{: .text-justify}

The outline will be updated in time as I post my drafts. 

The content in this post is heavily based on [^Kingma], [^Doersch], [^BleiStats], [^BishopBook], [^MurphyBook] and [^Goodfellow]. I also refer to [^Bounded] as a related material on which you could check for completeness.

# Intro
Before start talking about Variational Inference (VI) itself, I would like to point out to [Appendix A](/2019-11-11-vi-mean-field/#appendix-a) if you would like to know where it comes from. 
{: .text-justify}

Remember that in the last post we assume a latent variable model (LVM), where the hidden variables might be seen as the "cause" of our data, do not confuse it with *causality*. So we would like to find the posterior distribution, which is intractable because $$ p(\mathbf{z} \vert \mathbf{x})=\frac{p(\mathbf{x} \vert \mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})} $$, and the normalizing $$ p(\mathbf{x}) $$ term is intractable, that is, we might not have access to the distribution of $$ \mathbf{x} $$, or computing the integral to marginalizing $$ \mathbf{z} $$ is computationally infeasible. 
{: .text-justify}

Now, the core of VI is to approximate $$ p(\mathbf{z} \vert \mathbf{x}) $$ by another distribution, $$ q(\mathbf{z}) $$ of a family $$ \mathbf{Q} $$ of distributions, which we will define such that it has a tractable distribution, i.e., it has an analytical closed-form. We consider a restricted family of distributions $$ \mathbf{Q} $$ and then seek the member of this family for which the Kullback-Leibler (KL) divergence is minimized. Our goal is to restrict the family sufficiently that they comprise only tractable distributions, while at the same time allowing the family to be sufficiently rich and flexible that it can provide a good approximation to the true posterior distribution [^BishopBook]. 
{: .text-justify}

Each $$ q(\mathbf{z}) \in \mathbf{Q} $$ is a candidate approximation to the exact conditional. Our goal is to find the best candidate, the one closest in KL divergence to the exact conditional. The complexity of the family determines the complexity of this optimization. It is more difficult to optimize over a complex family than a simple family. Looking at the ELBO, the first term is an expected likelihood; it encourages densities that place their mass on configurations of the latent variables that explain the observed data. The second term is the negative divergence between the variational density and the prior; it encourages densities close to the prior. Thus the variational objective mirrors the usual balance between likelihood and prior. 
{: .text-justify}

The KL divergence is a way to compute the difference between distributions $$ p(\mathbf{z} \vert \mathbf{x}) $$ and $$ q(\mathbf{z}) $$, written as $$ D_{K L}\left(q(\mathbf{z}) \vert \vert p(\mathbf{z} \vert \mathbf{x})\right) $$. For a more in-depth discussion of the KL, see [Eric Jang](https://blog.evjang.com/2016/08/variational-bayes.html)’s and [Agustinus Kristiadi](https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/)'s posts. Now let’s do a bit of math on $$ D_{K L} $$:

$$ \begin{aligned} D_{K L}\left(q(\mathbf{z}) \vert \vert p(\mathbf{z} \vert \mathbf{x})\right) &=\int_{\mathbf{z}} q(\mathbf{z}) \log \left(\frac{q(\mathbf{z})}{p(\mathbf{z} \vert \mathbf{x})}\right) d\mathbf{z} \\ &=\int_{\mathbf{z}} q(\mathbf{z}) \log \left(\frac{q(\mathbf{z}) p(\mathbf{x})}{p(\mathbf{z}, \mathbf{x})}\right) d\mathbf{z} \\ &=\int_{\mathbf{z}} q(\mathbf{z})\left(\log p(\mathbf{x})+\log \frac{q(\mathbf{z})}{p(\mathbf{z}, \mathbf{x})}\right) d\mathbf{z} \\ &=\log p(\mathbf{x})+\mathbb{E}_{q(\mathbf{z})}\left[\log \frac{q(\mathbf{z})}{p(\mathbf{x} \vert \mathbf{z}) p(\mathbf{z})}\right] \\ &=\log p(\mathbf{x})+\mathbb{E}_{q(\mathbf{z})}\left[\log \frac{q(\mathbf{z})}{p(\mathbf{z})}-\log p(\mathbf{x} \vert \mathbf{z})\right] \end{aligned} $$

We can rewrite as
$$
\log p(\mathbf{x})-D_{K L}\left(q(\mathbf{z}) \vert \vert p(\mathbf{z} | \mathbf{x})\right)=\mathbb{E}_{q(\mathbf{z}} \log p(\mathbf{x} | \mathbf{z})-D_{K L}\left(q(\mathbf{z}) \| p(\mathbf{z})\right)
$$. 
{: .text-justify}

The right-hand side of the above equation is what we want to maximize when learning the true distributions, maximizing the log-likelihood of generating real data and minimize the difference between the real and estimated posterior distributions. We can rearrange as 
{: .text-justify}

$$
\log p(\mathbf{x})=\underbrace{\mathbb{E}_{q(\mathbf{z})} \log p(\mathbf{x} \vert \mathbf{z})-D_{K L}\left(q(\mathbf{z}) \vert \vert p(\mathbf{z})\right)}_{\mathcal{L}(\mathbf{x})} + D_{K L}\left(q(\mathbf{z}) \vert p(\mathbf{z} \vert \mathbf{x})\right)
$$ 
{: .text-centered}

Where $$ \mathcal{L}(\mathbf{x}) $$ is the Evidence Lower Bound (**ELBO**), also known as variational lower bound. Now we can rewrite as:

$$
\begin{aligned} \mathcal{L}(\mathbf{x}) &=\log p(\mathbf{x})-D_{K L}\left(q(\mathbf{z}) \vert p(\mathbf{z} \vert \mathbf{x})\right) \\ & \leq \log p(\mathbf{x}) \end{aligned}
\tag{1}\label{eq:elbo}
$$ 
{: .text-centered}

Due to the non-negativity of the KL divergence $$ D_{K L}\left(q(\mathbf{z}) \vert p(\mathbf{z} \vert \mathbf{x})\right) \geq 0 $$, the ELBO is a lower bound on the log-likelihood of the data, Equation \eqref{eq:elbo}. So the KL divergence $$ D _ { K L } \left( q ( \mathbf { z } ) \vert p( \mathbf { z } \vert \mathbf { x } ) \right) $$ determines two "distances": 
- By definition, the KL divergence of the approximate posterior from the true posterior.
- The gap between the ELBO and the marginal likelihood $$ \log p(\mathbf{x}) $$ which is also called the tightness of the bound. The better $$ q ( \mathbf { z } ) $$ approximates the true (posterior) distribution $$ p( \mathbf { z } \vert \mathbf { x } ) $$, in terms of the KL divergence, the smaller the gap.
{: .text-justify}

Maximizing the ELBO will concurrently optimize the two things we care about:
- It will approximately maximize the marginal likelihood $$ p(\mathbf{x}) $$. This means that our generative model will become better.
- It will minimize the KL divergence of the approximation $$ q(\mathbf{z}) $$ from the true posterior $$ p(\mathbf{z} \vert \mathbf{x}) $$, so $$ q(\mathbf{z}) $$ becomes better.
{: .text-justify}


# The Mean-Field Approximation (MF)
This approximation assumes that the posterior can be a factorized in a $$ m $$ disjoint groups resulting in an approximation of the form $$ q(\mathbf{z})=\prod_{m} q_{m}\left(\mathbf{z}^{m}\right) $$, where $$ m \leq d$$ and $$ d $$ is the dimension of $$ \mathbf{z} $$. In its fully factorized form, we assume that all dimensions of our latent variables are independent. Our goal is to solve this optimization problem $$ \min _{q_{1}, \ldots, q_{d}} \mathbb{K} \mathbb{L}(q \| p) $$. 
{: .text-justify}

Recall the ELBO: 
{: .text-justify}

$$
\begin{aligned}
\mathcal{L}(\mathbf{x}) &= \mathbb{E}_{q(\mathbf{z})} \log p(\mathbf{x} \vert \mathbf{z})-D_{K L}\left(q(\mathbf{z}) \vert \vert p(\mathbf{z})\right)\\ 
&= -\mathbb{E}_{q(\mathbf{z})}\left[\log \frac{q(\mathbf{z})}{p(\mathbf{z})}-\log p(\mathbf{x} \vert \mathbf{z})\right] \\
&= -\mathbb{E}_{q(\mathbf{z})}\left[\log \frac{q(\mathbf{z})}{p(\mathbf{x},\mathbf{z})}\right]
\end{aligned}
$$
{: .text-center}

The derived term is called the (negative) variational free energy, $$ \mathcal{L}(\mathbf{z}) $$, and we can rewrite it following the derivations from Murphy[^MurphyBook] as:
{: .text-justify}

$$
\begin{aligned} 
\mathcal{L}(\mathbf{z}) &= -\mathbb{E}_{\prod_{i}^{m} q_{i}\left(\mathbf{z}^{i}\right)}\left[\log \frac{\prod_{i}^{m} q_{i}\left(\mathbf{z}^{i}\right)}{p(\mathbf{x},\mathbf{z})}\right] \\
&=\int_{\mathbf{z}^1, \cdots, \mathbf{z}^d} \prod_{i}^{m} q_{i}\left(\mathbf{z}^{i}\right)\left[\log p(\mathbf{x},\mathbf{z})-\log\prod_{k}^{m} q_{k}\left(\mathbf{z}^{k}\right)\right] d\mathbf{z}^1, \cdots, d\mathbf{z}^d\\ 
&=\int_{\mathbf{z}^1, \cdots, \mathbf{z}^d} \prod_{i}^{m} q_{i}\left(\mathbf{z}^{i}\right)\left[\log p(\mathbf{x},\mathbf{z})-\sum_{k}^{m} \log q_{k}\left(\mathbf{z}^{k}\right)\right] d\mathbf{z}^1, \cdots, d\mathbf{z}^d\\ 
\mathcal{L}(\mathbf{z}^j)&=\int_{\mathbf{z}^{j}} \int_{\mathbf{z}^{-j}} q_{j}\left(\mathbf{z}^{j}\right) \prod_{i \neq j} q_{i}\left(\mathbf{z}^{i}\right)\left[\log p(\mathbf{x},\mathbf{z})-\sum_{k}^{m} \log q_{k}\left(\mathbf{z}^{k}\right)\right] d\mathbf{z}^j d\mathbf{z}^i\\ 
&=\int_{\mathbf{z}^{j}} q_{j}\left(\mathbf{z}^{j}\right) \left(\int_{i \neq j} \prod_{i \neq j} \left[q_{i}\left(\mathbf{z}^{i}\right)\right] \log p(\mathbf{x},\mathbf{z})d\mathbf{z}^i\right) d\mathbf{z}^j\\ 
&-\int_{\mathbf{z}^{j}} q_{j}\left(\mathbf{z}^{j}\right) \left(\int_{i \neq j} \prod_{i \neq j} q_{i}\left(\mathbf{z}^{i}\right) \left[\sum_{k \neq j} \log q_{k}\left(\mathbf{z}^{k}\right)+ \log q_{j}\left(\mathbf{z}^{j}\right)\right] d\mathbf{z}^i \right)d\mathbf{z}^j\\
&=\int_{\mathbf{z}^{j}} q_{j}\left(\mathbf{z}^{j}\right) \left(\int_{i \neq j} \prod_{i \neq j} \left[q_{i}\left(\mathbf{z}^{i}\right)\right] \log p(\mathbf{x},\mathbf{z})d\mathbf{z}^i\right) d\mathbf{z}^j\\ 
&-\int_{\mathbf{z}^{j}} q_{j}\left(\mathbf{z}^{j}\right) \log q_{j}\left(\mathbf{z}^{j}\right) \underbrace{\left(\int_{i \neq j} \prod_{i \neq j} q_{i}\left(\mathbf{z}^{i}\right) \sum_{k \neq j} \log q_{k}\left(\mathbf{z}^{k}\right) d\mathbf{z}^i \right)}_{\text{constant w.r.t } j}d\mathbf{z}^j\\ 
&=\int_{\mathbf{z}^{j}} q_{j}\left(\mathbf{z}^{j}\right) \log f_{j}\left(\mathbf{z}^{j}\right) d\mathbf{z} -\int_{\mathbf{z}^{j}} q_{j}\left(\mathbf{z}^{j}\right) \log q_{j}\left(\mathbf{z}^{j}\right) d\mathbf{z} + \text { constant }
\end{aligned}
$$

Jumping from the 4th to the 5th and 6th lines is a bit tricky. But the idea is to compute the (negative) variational free energy for each dimension, thus $$ \mathbf{z}^{-j} $$ means all other groups excluding $$ j $$. This can be done because multi-dimensional integral can be thought of as arbitrally summations which commutes in a global sum, you can see some nice Python code on [Will Wolf ](http://willwolf.io/2018/11/23/mean-field-variational-bayes/)'s blog about this. Since we can choose the order of the summation, we can derive MF formulas for a particular $$ j $$, and the other values depending on $$ i \neq j $$ are treated as constant, that is why only the term $$ \log q_j(\mathbf{z}^{j}) $$ remains from the inner integral in the right term on the 9th line from line 8. Moving forward, we have that:
{: .text-justify}

$$
\begin{equation}
\log f_{j}\left(\mathbf{z}^{j}\right) \triangleq \int_{\mathbf{z}^{-j}} \prod_{i \neq j} q_{i}\left(\mathbf{z}^{i}\right) \log p(\mathbf{x},\mathbf{z}) d\mathbf{z}^j =\mathbb{E}_{-q_{j}}[\log p(\mathbf{x},\mathbf{z})]
\tag{2}\label{eq:log-f}
\end{equation}
$$
{: .text-center}

It is just an expectation across all variables except for $$ j $$. Then 

$$
\begin{aligned} 
\mathcal{L}(\mathbf{z}^j) &=\int_{\mathbf{z}^{j}} q_{j}\left(\mathbf{z}^{j}\right) \log f_{j}\left(\mathbf{z}^{j}\right) d\mathbf{z}^j -\int_{\mathbf{z}^{j}} q_{j}\left(\mathbf{z}^{j}\right) \log q_{j}\left(\mathbf{z}^{j}\right) d\mathbf{z}^j +\text { const }\\
&= \int q_{j}(\mathbf{z}^{j}) \log \frac{f_{j}(\mathbf{z}^{j})}{q_{j}(\mathbf{z}^{j})} d\mathbf{z}^j\\
&= - \int q_{j}(\mathbf{z}^{j}) \log \frac{q_{j}(\mathbf{z}^{j})}{f_{j}(\mathbf{z}^{j})} d\mathbf{z}^j\\
&= - D_{KL}(q_{j} \vert\vert f_{j})
\end{aligned}
$$

We can maximize $$ \mathcal{L}(\mathbf{z}^j) $$ by minimizing the above $$ D_{KL}(q_{j} \vert\vert f_{j})$$. Note that $$ \log f_{j}(\mathbf{z}^{j}) = \mathbb{E}_{-q_{j}}[\log p(\mathbf{x},\mathbf{z})] $$ is an unnormalized log-likelihood written as a function of $$ \mathbf{z}^j $$, thus the ELBO reaches the minimum when $$ q_j = \mathbb{E}_{-q_{j}}[\log p(\mathbf{x},\mathbf{z})] $$: 

$$
\begin{equation} 
q_{j}(\mathbf{z}_{j}) =\frac{1}{Z_{j}} \exp (\mathbb{E}_{-q_{j}}[\log p(\mathbf{x},\mathbf{z})])\tag{3}\label{eq:qj}
\end{equation}
$$
$$
\begin{equation} 
\log q_{j}(\mathbf{z}_{j})=\mathbb{E}_{-q_{j}}[\log p(\mathbf{x},\mathbf{z})] + \text{constant}\tag{4}\label{eq:log-qj}
\end{equation}
$$

We can ignore the constant $$ Z_{j} $$ because we can normalize it after the optimization, saving us time of computing it. The functional form of the $$ q_{j} $$ distributions will be determined by the type of variables $$\mathbf{z}$$. During optimization, each $$ q_{j} $$ is independent, but we need to compute $$ q_{-j} $$ (for every $$ i \neq j $$) because of Equation \eqref{eq:log-f}, so the MF is an interative algorithm:
{: .text-justify}

1. Initialize the parameters of each $$ q_{j} $$ function.
2. For each $$ q_{j} $$, compute Equation \eqref{eq:log-qj}, where $$ q_{-j} $$ are constants.
3. Repeat until the convergence criteria is met.

There are a couple of good examples out there, I would ending up copying and pasting them here, so instead, I will refer you to great sources where you could check. Since I took some of the maths here from Murphy[^MurphyBook]'s, it is natural I would recommend checking subsection 21.3.2 and section 21.5 from this great book. Both [Wikipedia](https://en.wikipedia.org/wiki/Variational_Bayesian_methods#A_basic_example)'s page and [Brian Keng](https://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/)'s blog post have a similar formulation of MF for a Gaussian-gamma distribution inspired by Bishop[^BishopBook]'s subsection 10.1.3. Finally, one could also consider in checking in the tiny chapter written by [Rui Shu](http://ruishu.io/2018/02/28/mean-field-notes/).
{: .text-justify}


# Limitations of MF
The MF approximation is expressive because it can capture any marginal density of the latent variables, besides having closed updates for each $$ q_j $$. However, it assumes that $$ m $$ factors are independent of each other, thus it can not capture the correlation between them[^BleiStats]. Let's look at an example taken from Blei et al.[^BleiStats], Consider a highly correlated two-dimensional Gaussian distribution, Figure 1, which has an elongated shape.
{: .text-justify}

<figure>
 <div style="text-align:center;">
 <img style="margin-right:20;" src='/img/posts/bayes/bayes_1/mean_f.png' width="500"/>
 <p style="text-align:center;margin-top:0;"><b>Figure 1:</b> Taken from Blei et al.<sup id="fnref:BleiStats"><a href="#fn:BleiStats" class="footnote">3</a></sup></p>
 </div>
</figure>
{::options parse_block_html="true" /} 

Since the MF models independent variables, the optimal MF variational approximation to this posterior is a product of two Gaussian distributions. The approximation has the same mean as the original density, but the covariance structure is decoupled. The KL divergence penalizes more when a distribution $$ q $$ has more mass where $$ p $$ does not then when $$ p $$ has more mass than $$ q $$. Classical MF had its importance at the beginning, but it has multiple limitations in terms of modern and scalable applications[^Zhang], which were first tackled by a combination of VI stochastic optimization and distributed computing called Stochastic Variational Inference (SVI), which I will be discussing in the next post of this series. 
{: .text-justify}

# Appendix A
The content of this Appendix was taken from Bishop[^BishopBook]. Standard calculus is concerned with finding derivatives of functions. We can think of a function as a mapping that takes the value of a variable as input and returns the value of the function as the output. The derivative of the function then describes how the output value varies as we make infinitesimal changes to the input value. Similarly, we can define a functional as a mapping that takes a function as the input and that returns the value of the functional as the output. We can introduce the concept of a functional derivative, which expresses how the value of the functional changes in response to infinitesimal changes to the input function. Many problems can be expressed in terms of an optimization problem in which the quantity being optimized is a functional. 
{: .text-justify}

The solution is obtained by exploring all possible input functions to find the one that maximizes, or minimizes, the **functional**. Variational methods lend themselves to finding approximate solutions by restricting the range of functions over which the optimization is performed. In the context of VI, we need to find a functional $$q(\mathbf{z})$$ that maximizes the ELBO, thus *variational inference*. You can read more about variational calculus on this great Brian Keng’s blog [post](https://bjlkeng.github.io/posts/the-calculus-of-variations/) post.
{: .text-justify}


# References
[^Kingma]: Kingma, Diederik P. *Variational inference & deep learning: A new synthesis.* (2017). 
[^Doersch]: Doersch, Carl. *Tutorial on variational autoencoders.* arXiv preprint arXiv:1606.05908 (2016). 
[^Goodfellow]: Bengio, Yoshua, Ian Goodfellow, and Aaron Courville. Deep learning. Vol. 1. MIT press, 2017. 
[^BleiStats]: Blei, David M., Alp Kucukelbir, and Jon D. McAuliffe. *Variational inference: A review for statisticians.* Journal of the American Statistical Association 112.518 (2017): 859-877. 
[^Zhang]: Zhang, C., Butepage, J., Kjellstrom, H., & Mandt, S. *Advances in variational inference*. IEEE transactions on pattern analysis and machine intelligence (2018).
[^BishopBook]: Bishop, Christopher *M. Pattern recognition and machine learning*. Springer, 2006. 
[^MurphyBook]: Murphy, Kevin P. *Machine learning: a probabilistic perspective*. MIT press, 2012. 
[^Bounded]: Bounded Rationality, [Variational Bayes and The Mean-Field Approximation](https://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/).