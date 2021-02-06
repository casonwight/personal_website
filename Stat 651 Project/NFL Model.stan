data {
  int<lower=0> n;
  int<lower=0> num_teams;
  matrix[n,num_teams] y;
  matrix[n,num_teams] x;
  matrix[n,num_teams] ylag1;
  vector[n] t;
  real mu_a;
  real sigma_a;
  real mu_b;
  real sigma_b;
  real mu_theta;
  real sigma_theta;
  real a_sigma;
  real b_sigma;
  real lambda;
  real eta;
}


parameters {
  real overall_alpha;
  real overall_beta;
  real theta;
  real<lower=0> sigma;
  vector[num_teams] alphas;
  vector[num_teams] betas;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  for(i in 1:n){
    for(j in 1:num_teams){
      y[i,j] ~ normal(alphas[j] + betas[j] * ylag1[i,j] + theta * x[i,j], sigma);
    }
  }
  alphas ~ normal(overall_alpha, lambda);
  betas ~ normal(overall_beta, eta);
  overall_alpha ~ normal(mu_a, sigma_a);
  overall_beta ~ normal(mu_b, sigma_b);
  theta ~ normal(mu_theta, sigma_theta);
  sigma ~ gamma(a_sigma, b_sigma);
}

generated quantities {
  vector[n*num_teams] log_lik;
  for(i in 1:n){
    for(j in 1:num_teams){
      log_lik[(i-1)*num_teams + j] = normal_lpdf(y[i,j] | alphas[j] + betas[j] * ylag1[i,j] + theta * x[i,j], sigma);
    }
  }
}
