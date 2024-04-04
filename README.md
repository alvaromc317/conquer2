
<!-- README.md is generated from README.Rmd. Please edit that file -->

# conquer2

<!-- badges: start -->

[![License: GPL
v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Package
Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://cran.r-project.org/package=yourPackageName)
<!-- badges: end -->

conquer2 is merely an edited fork of the package `conquer`. The package
`conquer` is a wonderfully fast package for solving quantile regression
models that enforces the existence of an intercept. This fork of the
package merely removes the intercept and solves no-intercept models. For
any references or information regarding how the package works we refer
you to the original [repository](https://github.com/XiaoouPan/conquer)
and
[manual](https://cran.r-project.org/web/packages/conquer/conquer.pdf)

## Installation

You can install the development version of conquer2 from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("alvaromc317/conquer2")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(conquer2)

set.seed(123)

n = 500
p = 10
p2 = 5

X = matrix(rnorm(n * p, mean=2), ncol = p)
beta = c(rep(0, (p-p2)), 1:p2)
eps = rnorm(n, sd = 0.3)
y = X %*% beta + eps

model = conquer(X, y, tau=0.5)
```

    #> # A tibble: 10 × 3
    #>    `True coef` `Conquer coef` `Quantreg coef`
    #>          <dbl>          <dbl>           <dbl>
    #>  1           0           0               0   
    #>  2           0          -0.01           -0.03
    #>  3           0          -0.02           -0.01
    #>  4           0           0.01            0.02
    #>  5           0           0               0.02
    #>  6           1           1.02            1.02
    #>  7           2           2.01            2   
    #>  8           3           3               2.99
    #>  9           4           4.01            3.99
    #> 10           5           4.99            4.99
