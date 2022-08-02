# Reproducible Sign Rates

This package is designed to analyze data from the following setting.

- We are interested in real-valued parameters $\theta_1,\theta_2 ... \theta_n$.  For example, $\theta_i$ might indicate the effect of a particular drug on a particular gene.
- We have designed an experimental procedure which can estimate these parameters.
- We have two performed the experiment twice.

Given data of this form, we assume the user can calculate three quantities of interest for each parameter $\theta_i$.

1. Use the first replicate ("the training replicate") to produce a $p$-value, denoted $\rho_i$, for null hypothesis that $\theta_i=0$.
1. Use the training experiment to produce an object $\hat Y_i$ estimating the sign of $\theta_i$ (i.e., if the estimator is accurate, $\hat Y_i=1$ if $\theta_i>0$ and $\hat Y_i=-1$ if $\theta_i<0$).
1. Use the second replicate ("the validation replicate") produce an independent estimate $Y_i$ estimating the sign of $\theta_i$.

This package uses $\rho,\hat Y,Y$ to visualize whether the two replicates yielded the same results.  It does so using a quantity called the **Reproducible Sign Proportion**, defined as

$$\mathrm{RSP}(Y;\hat Y,\rho,\alpha) \triangleq \frac{|\{i:\ \hat Y_i = Y_i,\ \rho_i\leq \alpha\}|}{|{i:\ \rho_i\leq \alpha\}|}$$

In some cases each experimental procedure includes many subexperiments, and each subexperiment is approximately independent.  With subexperiments, this package can be used to also estimate confidence interval for the **Reproducible Sign Rate**, defined as $\mathrm{RSR}(\hat Y,\rho,\alpha)\triangleq \mathbb{E}_Y[\mathrm{RSP}(Y;\hat Y,\rho,\alpha)]$.

Example usage can be found in [this notebook](example_usage.ipynb).  API documentation can be found [here](https://prob-ml.github.io/reproducible-sign-rates/).  The package can be installed via `pip install reproduciblesignrates`.
