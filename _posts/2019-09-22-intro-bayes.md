---
layout: post
title: Introduction to Bayesian Approximate Inference
subtitle: Where it all begins
tags: [bayes-series]
---

# Before diving into the goodies

I have been learning about Bayesian Inference (BI), especially variational, in the past months and I decided to start a series of posts sharing what I have learned. Although there are several blog posts available (and I read a bunch of them), I might be able to bring another perspective on BI. Such a perspective I developed by reading from several sources, and I hope my writing will help those who are starting and those who want to do a quick review without going through all the sources I did (which is not a bad idea if you want to dive into the nuts and guts of BI). 
{: .text-justify}

In this and future posts, I will be referencing at the bottom of the page what sources I am using, and when I find necessary I will be referencing on the text as a link to a video, blog or book. I also will be explicitly saying from where the content is based, if necessary. As an example, this post is heavily based on Murphy[^MurphyBook] and Bishop[^BishopBook], besides others (see bottom of the page [^Stefano]  [^Aditya] [^BrianKeng]).
{: .text-justify}

Here is the outline of the blog posts of this series:
1. Introduction (you are here).
<!-- 2. Variational Inference - VI (I) : Intro & Mean-Field Factorization. -->

The outline will be updated in time as I post my drafts.

# Brief introduction
First thing I would say in an introductory post on bayesian approximate inference is to recall the general definition of the Bayes' theorem  
{: .text-justify}
$$
\begin{equation}
p(A \vert B) = \frac{p(B \vert A) \times p(A)}{p(B)} \tag{1}\label{eq:bayes-theo}
\end{equation}
$$
{: .text-center}

Let’s suppose our data, $$ \mathcal{D} $$, is a set of samples from a distribution ($$ p_{\mathcal{D}} $$). We want to use the data we already have to acquire some beliefs in order to apply when new data comes. Giving a task, e.g., classification and regression, we summarize these beliefs on a model. There are two main classifications of a model, either **generative** or **discriminative**. The first is used in statistical classification, it models a *decision boundary*, a conditional probability, between classes; the latter explicitly models the joint distribution of $$ \mathcal{D} $$ and our target set of variables, i.e., a generative model can be used to learn latent variables, classes, values of a regressor and much more. In other words, a discriminative tries to model $$ p(y \vert x)$$, and a generative $$ p(x,y)$$.  
{: .text-justify}

Thus a typical "query" we would do to a discriminative would be "Once you learned to classify animals, what is the animal in a given picture?", in the other hand, a generative could answer that and more questions like "Now you know what animals are, give me a picture with an animal $$ y $$", "Give me a lower-dimensional representation of this image" and so on. It doesn't mean that a discriminative couldn't answer the two previous questions, the main difference is that a generative model tries to learn a joint probability, the discriminative usually learns a mapping, which in most cases has worse performance than a generative for the same task. In this and following posts, we will be focusing on generative models.  
{: .text-justify}

A generative model tries to approximate the data distribution, $$ p_{\mathcal{D}} $$, given access to the dataset. We hope that if we are able to **learn** a good generative model, we can use the learned model for **inference**. In most models we are interested in summarizing all the information about the dataset $$ \mathcal{D} $$ in a finite set of parameters, thus we call them **parametric** models. So the task of learning is "discovering" the parameters of a model that would maximize the probability of $$ \mathcal{D} $$, and inference would be using the trained model to *infer* something, e.g, class value, lower-dimensional representation, continuous value for a regression problem, etc.  
{: .text-justify}

In a probabilistic view, our model is defined by its parameters and a distribution which might or might not have an analytical closed-form. Having a closed-form, we know exactly what are the parameters and what they represent, e.g., univariate Gaussian with parameters $$ \mu $$ and $$ \sigma $$. For instance, if we are trying to model a Gaussian distribution, then, following Equation \eqref{eq:bayes-theo}, we would have  
{: .text-justify}

$$
\begin{equation}
p(\theta \vert \mathcal{D}) = \frac{p(\mathcal{D} \vert \theta) \times p(\theta)}{p(\mathcal{D})} \quad \equiv \quad \text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}\tag{2}\label{eq:posteior-theta}
\end{equation}
$$
{: .text-center}

where $$ \theta = [\mu, \sigma] $$ is our set of parameters, which gives a hint on how we learn the parameters of our model. The likelihood $$ p(\mathcal{D} \vert \theta) $$ in Equation \eqref{eq:posteior-theta} expresses how probable the observed data set is for different settings of the parameter vector $$ \theta $$, whose integral with respect to $$ \theta $$ does not (necessarily) equal one, which means that $$ p(\mathcal{D} \vert \theta) $$  is not a probability distribution over $$ \theta $$.  
{: .text-justify}

Parametric models scale more efficiently (than non-parametric ones) with large datasets but are limited in the family of distributions they can represent, in the case of having a closed-form. That is, we assume that our model models a certain distribution, e.g., Gaussian, Categorical, Bernoulli, etc. Learning the parameters of a generative model means choosing such parameters within a family of distributions that minimizes some notion of distance between the model distribution and the data distribution.
{: .text-justify}

As said before, generative models learn a joint distribution over the entire data and some target set of variables, data and labels or latent variables, for instance. We can list three fundamental inference queries for evaluating a generative model:
  - Density estimation: Given a data point $$ x $$, what is $$ p_{\mathbf{\theta}}(\mathbf{x}) $$.
  - Sampling: Generation of novel data from the model distribution $$ x \sim p_{\mathbf{\theta}}(\mathbf{x}) $$.
  - Unsupervised representation learning: Learning meaningful feature representations for a datapoint $$ x $$.
{: .text-justify}


# Maximizing the probability of the data in our model
Okay, but what exactly means to *maximize the probability* of $$ \mathcal{D} $$ in our model? Suppose our model has a set of parameters $$ \theta $$ and it models a probability distribution from the family $$ \mathcal{P} $$, which has analytical form, thus know and we can evaluate several metrics. We can set up a distribution $$p \in \mathcal{P} $$ with parameters $$ \theta $$ and evaluate the probability of our data in $$ p $$. Even though we don't believe our data follows precisely the form of $$ \mathcal{P} $$, in real life we can accept a bit of modeling error.  
{: .text-justify}

For instance, suppose we have a dataset of values in the range $$ [10, 30] \in \mathbb{R}$$, and we know it follows a Gaussian distribution, where most of our data is in $$ [18, 21] $$. If we were given two models (a distribution and its parameters) and asked to choose one of them, we would choose the one that maximizes the probability of the data in that model. In this example, **Figure 1** shows the model that maximizes the probability, then we would stick with it. As you can imagine, in a real situation we don't know the parameters of our model that would maximize it, so is our job to make our model learn them.
{: .text-justify}

<figure>
    <div style="text-align:center;">
       <div style="display: inline-block;">
        <img style="margin-right:20;" src='/img/posts/bayes/bayes_0/gauss_1.svg' width="500"/>
        <p style="margin-top:0;"><b>Figure 1</b></p>
      </div>
      <div style="display: inline-block;">
        <img style="margin-right:20;" src='/img/posts/bayes/bayes_0/gauss_2.svg' width="500" />
        <p style="margin-top:0;"><b>Figure 2</b></p>
      </div>
    </div>
</figure>
{::options parse_block_html="true" /} 

In general, the probability of our model fit our data properly is $$ p_{\mathbf{\theta}}(\mathbf{x}) $$, where $$ \mathbf{\theta} $$ represents the model parameters and $$ \mathbf{x} $$ represents our data. A popular estimate for $$ \mathbf{\theta} $$ is the Maximum Likelihood Estimate (MLE), where it tries to find $$ \mathbf{\theta} $$ that maximized the likelihood function 
$$ p(\mathbf{x} \vert \theta) $$. In both the Bayesian and [frequentist](https://en.wikipedia.org/wiki/Frequentist_probability) paradigms, the likelihood function plays a central role. In a frequentist setting, $$ \mathbf{\theta} $$ is considered to be a fixed parameter, whose value is determined by some form of estimator, that is, this setting provides a point estimate (a single value) of the parameter (or sometimes a confidence interval). In the other hand, Bayesian setting provides a proper probability distribution for the parameter via the posterior distribution over $$ \mathbf{\theta} $$. MLE corresponds to choosing the **value** of $$ \mathbf{\theta} $$ for which the probability of the observed data set is maximized:  
{: .text-justify}

$$
\mathbf{\theta} = \underset{\mathbf{\theta}}{\arg \max } \prod_{i} p\left(x_{i} | \mathbf{\theta}\right)
$$

Another estimate widely used is called Maximum A Posteriori (MAP), which includes prior knowledge of $$ \mathbf{\theta} $$ when trying to find our parameters:  
{: .text-justify}

$$
\mathbf{\theta} = \underset{\mathbf{\theta}}{\arg \max } \prod_{i} p\left(x_{i} | \mathbf{\theta}\right) p(\mathbf{\theta})
$$


# Latent Variable Models

<div style="display: inline-block;text-align:center;">
  <img src='/img/posts/bayes/bayes_0/latent.svg' width="300" />
  <p style="text-align:center;margin-top:0;"><b>Figure 3</b> Graphical representation of an LVM. Where  $$ z $$ is a white node representing a hidden variable and $$ x $$ the observable variable (grey).
  Continuous arrow means direct dependency, thus $$ x $$ depends on $$ z $$. Dashed arrow means we can observe $$ z $$ undrectly, that is, 
  once we know $$ x $$ we might be able to infer $$ z $$ somehow.
  </p>
</div>

We may assume that the observed variables $$ \mathbf{x} $$ are correlated because they arise from a hidden common “cause”. A model with *hidden variables* are also known as **latent variable models** (LVMs), denoted as $$ \mathbf{z} $$, see **Figure 3**. In this setting, our generative model is the joint of our data and hidden variables. So, defining a joint distribution over observed and latent variables, $$ p_{\mathbf{\theta}}(\mathbf{x},\mathbf{z}) $$, the corresponding distribution of the observed variables alone is obtained by marginalization, $$ \int_{z} p_{\mathbf{\theta}}(\mathbf{x},z) dz $$. The introduction of latent variables thereby allows complicated distributions to be formed from simpler components[^BishopBook], because $$ p_{\mathbf{\theta}}(\mathbf{x},\mathbf{z}) = p_{\mathbf{\theta}}(\mathbf{x} \vert \mathbf{z}) p(\mathbf{z})$$, where the conditional can be a tractable distribution and the prior an arbitrary complex and intractable with no analytical form, then the joint is a complex distribution. Some applications of LVMs relies on discovering the hidden space and then computing a lower-dimensional representation of our data. For instance, suppose we have a dataset of images, where each has $$ N \times M $$ pixels, then our data lies on $$ \mathbb{R}^{N \times M} $$, which usually is an incredibly high dimensional space. We could use  a generative LVM in order to allow us to discover the latent variable associated with each image and recover a $$ L \times K $$ representation of them, where $$ L \ll N$$ and $$ K \ll M$$ (usually $$ L = 1$$).  
{: .text-justify}

Besides, some LVMs assumptions often have fewer parameters than models that directly represent correlation in the visible space; and the hidden variables in an LVM can serve as a **bottleneck**, which computes a compressed representation of the data, once we compute a lower-dimensional representation of the data and can use it to other tasks. This forms the basis of unsupervised learning. The latent variables do not need to (or might not) have any meaning, we might simply introduce latent variables in order to make the model more powerful[^MurphyBook], of course, it depends on the modeling and what is our tasks, stacking layers of latent variables doesn't necessarily implies a higher performance or generalization.
{: .text-justify}

A central task in the application of probabilistic models is the evaluation of the posterior distribution $$ p(\mathbf{z}\vert\mathbf{x}) $$ of the latent variables given the observed (visible) data variables, and the evaluation of expectations computed with respect to this distribution. The model might also contain some deterministic parameters or it may be a fully Bayesian model in which any unknown parameters are given prior distributions. For instance, suppose our model $$ p_{\mathbf{\theta}}(\mathbf{x}) $$ models a simple Gaussian, then $$ \mathbf{\theta} = [\boldsymbol{\mu},\boldsymbol{\sigma}] $$ and $$ p_{\mathbf{\theta}}(\mathbf{x})  \equiv \mathcal{N}(\mathbf{x} ; \boldsymbol{\mu},\boldsymbol{\sigma})$$. The parameters may come from other distributions, i.e., $$ \boldsymbol{\mu} \sim \pi(\boldsymbol{\mu}) $$ and $$ \boldsymbol{\sigma} \sim \tau(\boldsymbol{\sigma})$$, where $$ \pi $$ and $$ \tau $$ are also distributions.
{: .text-justify}

For many models of practical interest, it is infeasible to evaluate the posterior distribution $$ p(\mathbf{z}\vert\mathbf{x}) $$ or to compute expectations with respect to it. This could be because the dimensionality of the latent space is too high to work with or because the posterior distribution has a highly complex form for which expectations are not analytically tractable, a Neural Network, for instance.
{: .text-justify}

# Dealing with intractabilities

In the case of continuous variables, the required integrations may not have closed-form analytical solutions, while the dimensionality of the space and the complexity of the integrand may prohibit numerical integration. For discrete variables, the marginalizations involve summing over all possible configurations of the hidden variables, and though this is always possible in principle, we often find in practice that there may be exponentially many hidden states so that exact calculation is prohibitively expensive. That is, we can only see $$ \mathbf{x} $$, but we would like to infer the characteristics of $$ \mathbf{z} $$. Since our model specifies the joint distribution, we might want to find an approximation for
the posterior distribution $$ p(\mathbf{z}\vert\mathbf{x}) $$, which is
{: .text-justify}

$$
p(\mathbf{z} | \mathbf{x})=\frac{p(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})}
$$

The above equation is intractable because the data likelihood, $$ p(\mathbf{x}) $$, is not tractable:
$$
p(\mathbf{x})=\int p(\mathbf{z}) p(\mathbf{x} | \mathbf{z}) d \mathbf{z}
$$. Imagine that our data is a set of images in which we want to perform classification, we assumed an LVM because we think that the images have some related aspects ("causes") that we wish to consider in our modeling. The previous integral says that we need to sum all of the possible values of $$ \mathbf{z} $$, whatever they are, which is, of course, impossible.
{: .text-justify}

In such situations, we need to resort to approximation schemes, and these falls broadly into two classes, according to whether they rely on **stochastic** (sampling methods) or **deterministic** approximations. Stochastic techniques such as Markov Chain Monte Carlo (**MCMC**), which approximates inference via sampling, have enabled the widespread use of Bayesian methods across many domains. They generally have the property that given infinite computational resources, they can generate exact results, and the approximation arises from the use of a finite amount of processor time. In practice, sampling methods can be computationally demanding, often limiting their use to small-scale problems. Also, it can be difficult to know whether a sampling scheme is generating independent samples from the required distribution. 
{: .text-justify}

Deterministic approximation schemes are based on analytical approximations to the posterior distribution, for example by assuming that it factorizes in a particular way or that it has a specific parametric form such as a Gaussian. As such, they can never generate exact results, and so their strengths and weaknesses are complementary to those of sampling methods. We can cite **Laplace** approximation, which is based on a local Gaussian approximation to a mode (i.e., a maximum) of the distribution, and **variational inference** (VI) or variational Bayes (when inferring also the model's parameters). In the next post, I will be introducing VI concepts and the Mean Field Factorization method.
{: .text-justify}


# References
[^MurphyBook]: Murphy, Kevin P. *Machine learning: a probabilistic perspective*. MIT press, 2012.
[^BishopBook]: Bishop, Christopher *M. Pattern recognition and machine learning*. springer, 2006.
[^Stefano]: [Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/)
[^Aditya]: [Deep Generative Models](https://deepgenerativemodels.github.io/notes/)
[^BrianKeng]: [Bounded Rationality](https://bjlkeng.github.io/posts/normal-approximations-to-the-posterior-distribution/)