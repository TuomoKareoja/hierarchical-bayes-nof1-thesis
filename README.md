Hierarchical Bayesian Models for N-of-1 Trials
==============================

This repository the text and code for my statistics thesis applying Bayesian hierarchical models to analyze N-of-1 clinical trials.

N-of-1 trials in clinical setting are a way to systematize the common pattern of trying out multiple drugs after another to find one that has the an adequate balance between benefit and side-effects. This kind of trial and error strategy is common in managing long term diseases like high blood pressure and diabetes where the patient is expected to use the drugs for years or decades. What N-of-1 trials provide is framework statistically evaluate multiple drugs. The benefit from is over the common 'try until good enough' approach is that it is rigorous and can find the best drug among the candidates which is best for the patient, instead of the first one that is 'good enough'.

The downside of N-of-1 trials is that they take a lot of effort that only benefits one patient. If the approach would be used systematically for a larger group of patients the cost to benefit ratio would go down, but the problem of each test being separate from the others would remain. Compare this to the gold standard of clinical evaluations with random control trials (RCTs). In an RCT a group of patient is randomized into 2 or more groups which each group given a different treatment or acting as a control group. If one treatment is superior in this kind of test, we can claim that on average this is the best treatment for patients who are similar to the population that the patients in the study represent and could thus be used to inform treatment of a large group of users.

I argue that we can combine the benefits with of the N-of-1 method with the generalizability of RCTs. This can be done by applying hiearchical bayesian model to a group of N-of-1 studies. In this approach the results from the other patients can inform the results of the patient in question (if treatment A worked better than B for 90 % of other patients it is likely to work better for this patient too), without forcing the treatment recommendation to always match the one given to other patients (if after trying out treament A and B for this patient B worked significantly better this information, when strong enough, can inform the prior believe of the superiority of treatment A).

Key Features
------------

- Implementation of Bayesian models using PyMC3
- Simulation framework for creating realistic, but simple N-of-1 trial data
- Single-patient analysis with proper diagnostics
- Hierarchical models that allow intelligent pooling of information
- Visualization tools for posterior distributions and model checking

Technical Approach
------------

The models account for:
- Patient-specific treatment effects
- Natural disease progression (trend)
- Measurement error
- Population-level distributions of treatment effects

MCMC sampling with the No-U-Turn Sampler (NUTS) is used for posterior approximation, with detailed diagnostics to ensure sampling quality.
