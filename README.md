# 10701-f20-prog
Programming Homework of CMU's 10-701 Introduction to Machine Learning (PhD).

All solutions by Elvis Yan Pan (CMU 23')

## Homework 1
A Na&iuml;ve Bayes classifier with both discrete and continuous attributes.

Discrete attributes are modeled by categorical distribution with parameter <img src="https://render.githubusercontent.com/render/math?math=\alpha_{i,c,j}">, which shows the probability of the ith attribute being j given label c.

Continuous attributes are modeled by Gaussian distribution with parameter <img src="https://render.githubusercontent.com/render/math?math=\mu"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma^2">, where the probability is computed using
<img src="https://render.githubusercontent.com/render/math?math=P(x)=\frac{1}{\sqrt{2\pi(\sigma^2 %2B \epsilon)}}\exp\left(-\frac{(x-\mu)^2}{2(\sigma^2 %2B \epsilon)}\right).">

Run: `cd hw1` and then `python3 NaiveBayes.py`.
