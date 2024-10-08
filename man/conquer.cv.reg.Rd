% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/smqr.R
\name{conquer.cv.reg}
\alias{conquer.cv.reg}
\title{Cross-Validated Penalized Convolution-Type Smoothed Quantile Regression}
\usage{
conquer.cv.reg(
  X,
  Y,
  lambdaSeq = NULL,
  tau = 0.5,
  kernel = c("Gaussian", "logistic", "uniform", "parabolic", "triangular"),
  h = 0,
  penalty = c("lasso", "elastic", "group", "sparse-group", "scad", "mcp"),
  para.elastic = 0.5,
  group = NULL,
  weights = NULL,
  para.scad = 3.7,
  para.mcp = 3,
  kfolds = 5,
  numLambda = 50,
  epsilon = 0.001,
  iteMax = 500,
  phi0 = 0.01,
  gamma = 1.2,
  iteTight = 3
)
}
\arguments{
\item{X}{An \eqn{n} by \eqn{p} design matrix. Each row is a vector of observations with \eqn{p} covariates.}

\item{Y}{An \eqn{n}-dimensional response vector.}

\item{lambdaSeq}{(\strong{optional}) A sequence of candidate regularization parameters. If unspecified, the sequence will be generated by a simulated pivotal quantity approach proposed in Belloni and Chernozhukov (2011).}

\item{tau}{(\strong{optional}) Quantile level (between 0 and 1). Default is 0.5.}

\item{kernel}{(\strong{optional}) A character string specifying the choice of kernel function. Default is "Gaussian". Choices are "Gaussian", "logistic", "uniform", "parabolic" and "triangular".}

\item{h}{(\strong{optional}) The bandwidth parameter for kernel smoothing. Default is \eqn{\max\{0.5 * (log(p) / n)^{0.25}, 0.05\}}. The default will be used if the input value is less than or equal to 0.}

\item{penalty}{(\strong{optional}) A character string specifying the penalty. Default is "lasso" (Tibshirani, 1996). The other options are "elastic" for elastic-net (Zou and Hastie, 2005), "group" for group lasso (Yuan and Lin, 2006), "sparse-group" for sparse group lasso (Simon et al., 2013), "scad" (Fan and Li, 2001) and "mcp" (Zhang, 2010).}

\item{para.elastic}{(\strong{optional}) The mixing parameter between 0 and 1 (usually noted as \eqn{\alpha}) for elastic net. The penalty is defined as \eqn{\alpha ||\beta||_1 + (1 - \alpha) ||\beta||_2^2}. Default is 0.5.
Setting \code{para.elastic = 1} gives the lasso penalty, and setting \code{para.elastic = 0} yields the ridge penalty. Only specify it when \code{penalty = "elastic"}.}

\item{group}{(\strong{optional}) A \eqn{p}-dimensional vector specifying group indices. Only specify it if \code{penalty = "group"} or \code{penalty = "sparse-group"}.
For example, if \eqn{p = 10}, and we assume the first 3 coefficients belong to the first group, and the last 7 coefficients belong to the second group, then the argument should be \code{group = c(rep(1, 3), rep(2, 7))}. If not specified, then the penalty will be the classical lasso.}

\item{weights}{(\strong{optional}) A vector specifying groups weights for group Lasso and sparse group Lasso. The length must be equal to the number of groups. If not specified, the default weights are square roots of group sizes.
For example , if \code{group = c(rep(1, 3), rep(2, 7))}, then the default weights are \eqn{\sqrt{3}} for the first group, and \eqn{\sqrt{7}} for the second group.}

\item{para.scad}{(\strong{optional}) The constant parameter for "scad". Default value is 3.7. Only specify it if \code{penalty = "scad"}.}

\item{para.mcp}{(\strong{optional}) The constant parameter for "mcp". Default value is 3. Only specify it if \code{penalty = "mcp"}.}

\item{kfolds}{(\strong{optional}) Number of folds for cross-validation. Default is 5.}

\item{numLambda}{(\strong{optional}) Number of \eqn{\lambda} values for cross-validation if \code{lambdaSeq} is unspeficied. Default is 50.}

\item{epsilon}{(\strong{optional}) A tolerance level for the stopping rule. The iteration will stop when the maximum magnitude of the change of coefficient updates is less than \code{epsilon}. Default is 0.001.}

\item{iteMax}{(\strong{optional}) Maximum number of iterations. Default is 500.}

\item{phi0}{(\strong{optional}) The initial quadratic coefficient parameter in the local adaptive majorize-minimize algorithm. Default is 0.01.}

\item{gamma}{(\strong{optional}) The adaptive search parameter (greater than 1) in the local adaptive majorize-minimize algorithm. Default is 1.2.}

\item{iteTight}{(\strong{optional}) Maximum number of tightening iterations in the iteratively reweighted \eqn{\ell_1}-penalized algorithm. Only specify it if the penalty is scad or mcp. Default is 3.}
}
\value{
An object containing the following items will be returned:
\describe{
\item{\code{coeff.min}}{A \eqn{(p)} vector of estimated coefficients including the intercept selected by minimizing the cross-validation errors.}
\item{\code{coeff.1se}}{A \eqn{(p)} vector of estimated coefficients including the intercept. The corresponding \eqn{\lambda} is the largest \eqn{\lambda} such that the cross-validation error is within 1 standard error of the minimum.}
\item{\code{lambdaSeq}}{The sequence of regularization parameter candidates for cross-validation.}
\item{\code{lambda.min}}{Regularization parameter selected by minimizing the cross-validation errors. This is the corresponding \eqn{\lambda} of \code{coeff.min}.}
\item{\code{lambda.1se}}{The largest regularization parameter such that the cross-validation error is within 1 standard error of the minimum. This is the corresponding \eqn{\lambda} of \code{coeff.1se}.}
\item{\code{deviance}}{Cross-validation errors based on the quantile loss. The length is equal to the length of \code{lambdaSeq}.}
\item{\code{deviance.se}}{Estimated standard errors of \code{deviance}. The length is equal to the length of \code{lambdaSeq}.}
\item{\code{bandwidth}}{Bandwidth value.}
\item{\code{tau}}{Quantile level.}
\item{\code{kernel}}{Kernel function.}
\item{\code{penalty}}{Penalty type.}
\item{\code{n}}{Sample size.}
\item{\code{p}}{Number of covariates.}
}
}
\description{
Fit sparse quantile regression models via regularized conquer methods with "lasso", "elastic-net", "group lasso", "sparse group lasso", "scad" and "mcp" penalties. The regularization parameter \eqn{\lambda} is selected via cross-validation.
}
\examples{
n = 100; p = 200; s = 5
beta = c(rep(1.5, s), rep(0, p - s))
X = matrix(rnorm(n * p), n, p)
Y = X \%*\% beta + rt(n, 2)

## Cross-validated regularized conquer with lasso penalty at tau = 0.7
fit.lasso = conquer.cv.reg(X, Y, tau = 0.7, penalty = "lasso")
beta.lasso = fit.lasso$coeff.min

## Cross-validated regularized conquer with elastic-net penalty at tau = 0.7
fit.elastic = conquer.cv.reg(X, Y, tau = 0.7, penalty = "elastic", para.elastic = 0.7)
beta.elastic = fit.elastic$coeff.min

## Cross-validated regularized conquer with scad penalty at tau = 0.7
fit.scad = conquer.cv.reg(X, Y, tau = 0.7, penalty = "scad")
beta.scad = fit.scad$coeff.min

## Regularized conquer with group lasso at tau = 0.7
beta = c(rep(1.3, 2), rep(1.5, 3), rep(0, p - s))
err = rt(n, 2)
Y = X \%*\% beta + err
group = c(rep(1, 2), rep(2, 3), rep(3, p - s))
fit.group = conquer.cv.reg(X, Y,tau = 0.7, penalty = "group", group = group)
beta.group = fit.group$coeff.min
}
\references{
Belloni, A. and Chernozhukov, V. (2011). \eqn{\ell_1} penalized quantile regression in high-dimensional sparse models. Ann. Statist., 39, 82-130.

Fan, J. and Li, R. (2001). Variable selection via nonconcave regularized likelihood and its oracle properties. J. Amer. Statist. Assoc., 96, 1348-1360.

Fan, J., Liu, H., Sun, Q. and Zhang, T. (2018). I-LAMM for sparse learning: Simultaneous control of algorithmic complexity and statistical error. Ann. Statist., 46, 814-841.

Koenker, R. and Bassett, G. (1978). Regression quantiles. Econometrica, 46, 33-50.

Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2013). A sparse-group lasso. J. Comp. Graph. Statist., 22, 231-245.

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. J. R. Statist. Soc. Ser. B, 58, 267–288.

Tan, K. M., Wang, L. and Zhou, W.-X. (2022). High-dimensional quantile regression: convolution smoothing and concave regularization. J. Roy. Statist. Soc. Ser. B, 84, 205-233.

Yuan, M. and Lin, Y. (2006). Model selection and estimation in regression with grouped variables., J. Roy. Statist. Soc. Ser. B, 68, 49-67.

Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. Ann. Statist., 38, 894-942.

Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. J. R. Statist. Soc. Ser. B, 67, 301-320.
}
\seealso{
See \code{\link{conquer.reg}} for regularized quantile regression with a prescribed \eqn{lambda}.
}
