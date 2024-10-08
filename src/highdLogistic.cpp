# include <RcppArmadillo.h>
# include <cmath>
# include "basicOp.h"
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
double lossLogisticHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, const double tau, const double h, const double h1) {
  arma::vec res = Y - Z * beta;
  arma::vec temp = tau * res + h * arma::log(1.0 + arma::exp(-h1 * res));
  return arma::mean(temp);
}

// [[Rcpp::export]]
double updateLogisticHd(const arma::mat& Z, const arma::vec& Y, const arma::vec& beta, arma::vec& grad, const double tau, const double n1, const double h, 
                        const double h1) {
  arma::vec res = Y - Z * beta;
  arma::vec der = 1.0 / (1.0 + arma::exp(res * h1)) - tau;
  grad = n1 * Z.t() * der;
  arma::vec temp = tau * res + h * arma::log(1.0 + arma::exp(-h1 * res));
  return arma::mean(temp);
}

// [[Rcpp::export]]
double lammLogisticLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double phi, 
                         const double gamma, const int p, const double h, const double n1, const double h1) {
  double phiNew = phi;
  arma::vec betaNew(p);
  arma::vec grad(p);
  double loss = updateLogisticHd(Z, Y, beta, grad, tau, n1, h, h1);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    double fVal = lossLogisticHd(Z, Y, betaNew, tau, h, h1);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// [[Rcpp::export]]
double lammLogisticElastic(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, arma::vec& beta, const double tau, const double alpha, 
                           const double phi, const double gamma, const int p, const double h, const double n1, const double h1) {
  double phiNew = phi;
  arma::vec betaNew(p);
  arma::vec grad(p);
  double loss = updateLogisticHd(Z, Y, beta, grad, tau, n1, h, h1);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = alpha * Lambda / phiNew;
    betaNew = softThresh(first, second, p) / (1.0 + (2.0 - 2 * alpha) * Lambda / phiNew);
    double fVal = lossLogisticHd(Z, Y, betaNew, tau, h, h1);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// [[Rcpp::export]]
double lammLogisticGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, arma::vec& beta, const double tau, const arma::vec& group, 
                              const arma::vec& weight, const double phi, const double gamma, const int p, const int G, const double h, const double n1, 
                              const double h1) {
  double phiNew = phi;
  arma::vec betaNew(p);
  arma::vec grad(p);
  double loss = updateLogisticHd(Z, Y, beta, grad, tau, n1, h, h1);
  while (true) {
    arma::vec subNorm = arma::zeros(G);
    betaNew = beta - grad / phiNew;
    for (int i = 0; i < p; i++) {
      subNorm(group(i)) += betaNew(i) * betaNew(i);
    }
    subNorm = arma::max(1.0 - lambda * weight / (phiNew * arma::sqrt(subNorm)), arma::zeros(G));
    for (int i = 0; i < p; i++) {
      betaNew(i) *= subNorm(group(i));
    }
    double fVal = lossLogisticHd(Z, Y, betaNew, tau, h, h1);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// [[Rcpp::export]]
double lammLogisticSparseGroupLasso(const arma::mat& Z, const arma::vec& Y, const arma::vec& Lambda, const double lambda, arma::vec& beta, const double tau, 
                                    const arma::vec& group, const arma::vec& weight, const double phi, const double gamma, const int p, const int G, 
                                    const double h, const double n1, const double h1) {
  double phiNew = phi;
  arma::vec betaNew(p);
  arma::vec grad(p);
  double loss = updateLogisticHd(Z, Y, beta, grad, tau, n1, h, h1);
  while (true) {
    arma::vec first = beta - grad / phiNew;
    arma::vec second = Lambda / phiNew;
    betaNew = softThresh(first, second, p);
    arma::vec subNorm = arma::zeros(G);
    for (int i = 0; i < p; i++) {
      subNorm(group(i)) += betaNew(i) * betaNew(i);
    }
    subNorm = arma::max(1.0 - lambda * weight / (phiNew * arma::sqrt(subNorm)), arma::zeros(G));
    for (int i = 0; i < p; i++) {
      betaNew(i) *= subNorm(group(i));
    }
    double fVal = lossLogisticHd(Z, Y, betaNew, tau, h, h1);
    arma::vec diff = betaNew - beta;
    double psiVal = loss + arma::as_scalar(grad.t() * diff) + 0.5 * phiNew * arma::as_scalar(diff.t() * diff);
    if (fVal <= psiVal) {
      break;
    }
    phiNew *= gamma;
  }
  beta = betaNew;
  return phiNew;
}

// [[Rcpp::export]]
arma::vec logisticLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                        const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                            const double n1, const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, 
                            const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticElastic(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const double alpha, const int p, const double n1, 
                          const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                          const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaLasso(lambda, p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticElastic(Z, Y, Lambda, betaNew, tau, alpha, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticElasticWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const double alpha, 
                              const int p, const double n1, const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, 
                              const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticElastic(Z, Y, Lambda, betaNew, tau, alpha, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const arma::vec& group, 
                             const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1,
                             const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  arma::vec betaNew = beta;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticGroupLasso(Z, Y, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticGroupLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, 
                                 const arma::vec& group, const arma::vec& weight, const int p, const int G, const double n1, const double h, 
                                 const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                 const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticGroupLasso(Z, Y, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticSparseGroupLasso(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const arma::vec& group, 
                                   const arma::vec& weight, const int p, const int G, const double n1, const double h, const double h1, 
                                   const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticSparseGroupLasso(Z, Y, Lambda, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticSparseGroupLassoWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, 
                                       const arma::vec& group, const arma::vec& weight, const int p, const int G, const double n1, const double h, 
                                       const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = lambda * arma::ones(p);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticSparseGroupLasso(Z, Y, Lambda, lambda, betaNew, tau, group, weight, phi, gamma, p, G, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticScad(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                       const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                       const int iteTight = 3, const double para = 3.7) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 1;
  // Tightening
  arma::vec beta0(p);
  while (iteT <= iteTight) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaSCAD(beta, lambda, p, para);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      phi = lammLogisticLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticScadWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                           const double n1, const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, 
                           const double epsilon = 0.001, const int iteMax = 500, const double para = 3.7) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaSCAD(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticMcp(const arma::mat& Z, const arma::vec& Y, const double lambda, const double tau, const int p, const double n1, const double h, 
                      const double h1, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                      const int iteTight = 3, const double para = 3) {
  arma::vec beta = lasso(Z, Y, lambda, tau, p, n1, phi0, gamma, epsilon, iteMax);
  arma::vec quant = {tau};
  arma::vec betaNew = beta;
  // Contraction
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  int iteT = 1;
  // Tightening
  arma::vec beta0(p);
  while (iteT <= iteTight) {
    iteT++;
    beta = betaNew;
    beta0 = betaNew;
    Lambda = cmptLambdaMCP(beta, lambda, p, para);
    phi = phi0;
    ite = 0;
    while (ite <= iteMax) {
      ite++;
      phi = lammLogisticLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
      phi = std::max(phi0, phi / gamma);
      if (arma::norm(betaNew - beta, "inf") <= epsilon) {
        break;
      }
      beta = betaNew;
    }
    if (arma::norm(betaNew - beta0, "inf") <= epsilon) {
      break;
    }
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec logisticMcpWarm(const arma::mat& Z, const arma::vec& Y, const double lambda, const arma::vec& betaWarm, const double tau, const int p, 
                          const double n1, const double h, const double h1, const double phi0 = 0.01, const double gamma = 1.2, 
                          const double epsilon = 0.001, const int iteMax = 500, const double para = 3) {
  arma::vec beta = betaWarm;
  arma::vec betaNew = beta;
  arma::vec Lambda = cmptLambdaMCP(beta, lambda, p, para);
  double phi = phi0;
  int ite = 0;
  while (ite <= iteMax) {
    ite++;
    phi = lammLogisticLasso(Z, Y, Lambda, betaNew, tau, phi, gamma, p, h, n1, h1);
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

// [[Rcpp::export]]
arma::vec conquerLogisticLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                               const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::vec betaHat = logisticLasso(Z, Y, lambda, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat %= sx1;
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerLogisticLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, const double phi0 = 0.01, 
                                  const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::mat betaSeq(p, nlambda);
  arma::vec betaHat = logisticLasso(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = logisticLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.each_col() %= sx1;
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerLogisticElastic(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double alpha, const double h, 
                                 const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::vec betaHat = logisticElastic(Z, Y, lambda, tau, alpha, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat %= sx1;
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerLogisticElasticSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double alpha, const double h, 
                                    const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::mat betaSeq(p, nlambda);
  arma::vec betaHat = logisticElastic(Z, Y, lambdaSeq(0), tau, alpha, p, n1, h, h1, phi0, gamma, epsilon, iteMax);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = logisticElasticWarm(Z, Y, lambdaSeq(i), betaWarm, tau, alpha, p, n1, h, h1, phi0, gamma, epsilon, iteMax);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.each_col() %= sx1;
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerLogisticGroupLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const arma::vec& group, const arma::vec& weight, const int G, 
                                    const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::vec betaHat = logisticGroupLasso(Z, Y, lambda, tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat %= sx1;
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerLogisticGroupLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const arma::vec& group, const arma::vec& weight, const int G, 
                                       const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::mat betaSeq(p, nlambda);
  arma::vec betaHat = logisticGroupLasso(Z, Y, lambdaSeq(0), tau, group, weight, p, G, n1, h, h1, phi0, gamma, epsilon, iteMax);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = logisticGroupLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1, h, h1, phi0, gamma, epsilon, iteMax);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.each_col() %= sx1;
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerLogisticSparseGroupLasso(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const arma::vec& group, const arma::vec& weight, const int G, 
                                          const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                          const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::vec betaHat = logisticSparseGroupLasso(Z, Y, lambda, tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat %= sx1;
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerLogisticSparseGroupLassoSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const arma::vec& group, const arma::vec& weight, 
                                             const int G, const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                             const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::mat betaSeq(p, nlambda);
  arma::vec betaHat = logisticSparseGroupLasso(Z, Y, lambdaSeq(0), tau, group, weight, p, G, n1, h, h1, phi0, gamma, epsilon, iteMax);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = logisticSparseGroupLassoWarm(Z, Y, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1, h, h1, phi0, gamma, epsilon, iteMax);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.each_col() %= sx1;
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerLogisticScad(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                              const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::vec betaHat = logisticScad(Z, Y, lambda, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat %= sx1;
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerLogisticScadSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, 
                                 const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                                 const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::mat betaSeq(p, nlambda);
  arma::vec betaHat = logisticScad(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = logisticScadWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax, para);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.each_col() %= sx1;
  return betaSeq;
}

// [[Rcpp::export]]
arma::vec conquerLogisticMcp(const arma::mat& X, arma::vec Y, const double lambda, const double tau, const double h, const double phi0 = 0.01, 
                             const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, const int iteTight = 3, const double para = 3.0) {
  const int n = X.n_rows, p = X.n_cols;
  const double h1 = 1.0 / h;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::vec betaHat = logisticMcp(Z, Y, lambda, tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat %= sx1;
  return betaHat;
}

// [[Rcpp::export]]
arma::mat conquerLogisticMcpSeq(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const double tau, const double h, 
                                const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500, 
                                const int iteTight = 3, const double para = 3.0) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h, n1 = 1.0 / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  arma::mat betaSeq(p, nlambda);
  arma::vec betaHat = logisticMcp(Z, Y, lambdaSeq(0), tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaSeq.col(0) = betaHat;
  arma::vec betaWarm = betaHat;
  for (int i = 1; i < nlambda; i++) {
    betaHat = logisticMcpWarm(Z, Y, lambdaSeq(i), betaWarm, tau, p, n1, h, h1, phi0, gamma, epsilon, iteMax, para);
    betaSeq.col(i) = betaHat;
    betaWarm = betaHat;
  }
  betaSeq.each_col() %= sx1;
  return betaSeq;
}

// [[Rcpp::export]]
Rcpp::List cvLogisticLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                               const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p), betaWarm(p);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = logisticLasso(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = logisticLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = logisticLasso(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat %= sx1;
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = logisticLasso(Z, Y, lambdaSeq(seIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHatSe %= sx1;
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvLogisticElasticWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const double alpha, 
                                 const int kfolds, const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, 
                                 const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p), betaWarm(p);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = logisticElastic(trainZ, trainY, lambdaSeq(0), tau, alpha, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = logisticElasticWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, alpha, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = logisticElastic(Z, Y, lambdaSeq(cvIdx), tau, alpha, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat %= sx1;
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = logisticElastic(Z, Y, lambdaSeq(seIdx), tau, alpha, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHatSe %= sx1;
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvLogisticGroupLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                                    const arma::vec& group, const arma::vec& weight, const int G, const double h, const double phi0 = 0.01, const double gamma = 1.2, 
                                    const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p), betaWarm(p);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = logisticGroupLasso(trainZ, trainY, lambdaSeq(0), tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = logisticGroupLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = logisticGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat %= sx1;
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = logisticGroupLasso(Z, Y, lambdaSeq(seIdx), tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHatSe%= sx1;
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvLogisticSparseGroupLassoWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, 
                                          const int kfolds, const arma::vec& group, const arma::vec& weight, const int G, const double h, const double phi0 = 0.01, 
                                          const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p), betaWarm(p);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = logisticSparseGroupLasso(trainZ, trainY, lambdaSeq(0), tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = logisticSparseGroupLassoWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, group, weight, p, G, n1Train, h, h1, phi0, gamma, epsilon, iteMax);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = logisticSparseGroupLasso(Z, Y, lambdaSeq(cvIdx), tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHat %= sx1;
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = logisticSparseGroupLasso(Z, Y, lambdaSeq(seIdx), tau, group, weight, p, G, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax);
  betaHatSe %= sx1;
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvLogisticScadWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                              const double h, const double phi0 = 0.01, const double gamma = 1.2, const double epsilon = 0.001, const int iteMax = 500,
                              const int iteTight = 3, const double para = 3.7) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p), betaWarm(p);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = logisticScad(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = logisticScadWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, para);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = logisticScad(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat %= sx1;
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = logisticScad(Z, Y, lambdaSeq(seIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHatSe %= sx1;
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

// [[Rcpp::export]]
Rcpp::List cvLogisticMcpWarm(const arma::mat& X, arma::vec Y, const arma::vec& lambdaSeq, const arma::vec& folds, const double tau, const int kfolds, 
                             const double h, const double phi0 = 0.01, const double gamma = 1.5, const double epsilon = 0.001, const int iteMax = 500,
                             const int iteTight = 3, const double para = 3) {
  const int n = X.n_rows, p = X.n_cols, nlambda = lambdaSeq.size();
  const double h1 = 1.0 / h;
  arma::vec betaHat(p), betaWarm(p);
  arma::vec dev = arma::zeros(nlambda);
  arma::vec devsq = arma::zeros(nlambda);
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx1 = 1.0 / arma::stddev(X, 0, 0).t();
  arma::mat Z = standardize(X, mx, sx1, p); // No intercept column added
  for (int j = 1; j <= kfolds; j++) {
    arma::uvec idx = arma::find(folds == j);
    arma::uvec idxComp = arma::find(folds != j);
    double n1Train = 1.0 / idxComp.size();
    arma::mat trainZ = Z.rows(idxComp), testZ = Z.rows(idx);
    arma::vec trainY = Y.rows(idxComp), testY = Y.rows(idx);
    betaHat = logisticMcp(trainZ, trainY, lambdaSeq(0), tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
    betaWarm = betaHat;
    lossQr(testZ, testY, betaHat, tau, 0, dev, devsq);
    for (int i = 1; i < nlambda; i++) {
      betaHat = logisticMcpWarm(trainZ, trainY, lambdaSeq(i), betaWarm, tau, p, n1Train, h, h1, phi0, gamma, epsilon, iteMax, para);
      lossQr(testZ, testY, betaHat, tau, i, dev, devsq);
      betaWarm = betaHat;
    }
  }
  dev /= n;
  devsq = arma::sqrt(devsq - n * arma::square(dev)) / n;
  arma::uword cvIdx = arma::index_min(dev);
  betaHat = logisticMcp(Z, Y, lambdaSeq(cvIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHat %= sx1;
  arma::uword seIdx = arma::max(arma::find(dev <= dev(cvIdx) + devsq(cvIdx)));
  arma::vec betaHatSe = logisticMcp(Z, Y, lambdaSeq(seIdx), tau, p, 1.0 / n, h, h1, phi0, gamma, epsilon, iteMax, iteTight, para);
  betaHatSe %= sx1;
  return Rcpp::List::create(Rcpp::Named("coeff") = betaHat, Rcpp::Named("coeffSe") = betaHatSe, Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("lambdaSe") = lambdaSeq(seIdx), Rcpp::Named("deviance") = dev, Rcpp::Named("devianceSd") = devsq);
}

