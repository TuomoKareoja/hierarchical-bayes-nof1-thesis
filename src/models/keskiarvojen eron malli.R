# MALLIN SPESIFIOINTI 

#ladataan tarvittavat paketit

library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library(shinystan)
library(dplyr)

#muokataan data STANille sopivaan muotoon

data1 <- data %>% 
  group_by(Potilas, id_blokki, id_jakso, hoito_vai_plasebo) %>% 
  summarise(keskiarvo_jakso = mean(Tulos))

data1 <- data1 %>% 
  arrange(Potilas, id_blokki, hoito_vai_plasebo) %>% 
  group_by(Potilas, id_blokki) %>% 
  mutate(plasebo = lag(keskiarvo_jakso)) %>%
  filter(hoito_vai_plasebo==1) %>%
  mutate(ero = keskiarvo_jakso - plasebo)

data1 <- data1 %>%
  group_by(Potilas) %>%
  summarise(erojen_keskiarvo = mean(ero), erojen_keskivirhe = sd(keskiarvo_jakso)/n())

#Mallien spesifiointi


# malli jossa hajonta osittain tunnettu
nOf1 = "
data {
int<lower=0> N; //potilaiden m‰‰r‰
int<lower=2> J; //crossover jaksojen m‰‰r‰
vector[N] y; //hoitovasteiden ero blokkien sis‰ll‰ mean(a) - mean(b)
vector<lower=0>[N] s; //hoitovasteiden erojen keskivirhe potilaan sis‰ll‰
real<lower=0> S; //hoitovasteiden y:iden keskihajonta potilaiden v‰lill‰
}
parameters {
real mu; //populaation keskiarvo
real<lower=0> tau; //populaation keskihajonta
vector[N] theta; //potilastason erot hoitovasteessa
real<lower=0> ultVar; //tuntematon potilaan sis‰isen keskivirheen komponentti
}
transformed parameters {
vector<lower=0>[N] sigma; //potilastason varianssi
sigma = s * sqrt(ultVar);
}
model {
mu ~ normal(0, S*100); // heikon priorin varianssi suuri
tau ~ cauchy(0, 5); //ei voida k‰ytt‰‰ tavanomaista inverssi‰ gammajakaumaa!
ultVar ~  inv_chi_square(J);  // jos havaintoja olisi joillain potilailla enemm‰n kuin toisilla, niin t‰ss‰ voisi k‰ytt‰‰ harmonista keskiarvoa potilaiden havaintojen m‰‰r‰st‰, jolloin potilaiden sis‰isen heterogenian arviossa painottuisivat ne potilaat, joilta on enemm‰n havaintoja
theta ~ normal(mu, tau); //huomaa parametrien hierarkisuus!
y ~ normal(theta, sigma);
}
generated quantities {
vector[N] y_rep;
for(n in 1:N)
y_rep[n] = normal_rng(theta[n], sigma[n]);
}
"


#prosessoidaan malli
stanDso <- stan_model(model_code = nOf1)

# Syˆtet‰‰n data mallille ja lasketaan
dataList <- list(N = potilaiden_maara, J = blokkien_maara, y = data1$erojen_keskiarvo, s = data1$erojen_keskivirhe, S = sd(data1$erojen_keskiarvo))
stanFit <- sampling(object = stanDso, data = dataList, chains = 3, iter = 2000, warmup = 200)

# Luodaan erillinen vectori, jossa todelliset keskiarvot. T‰t‰ k‰ytet‰‰n testamaan sit‰, voiko malli generoida havaittua dataa muistuttavaa aineistoa
y <- data$erojen_keskiarvo

# Tarkastellaan mallin ominaisuuksia Shinystanin verkkosivun kautta
my_shinystan <- as.shinystan(stanFit)
launch_shinystan(my_shinystan)
summary(data1)
