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
int<lower=0> N; //potilaiden m??r?
int<lower=2> J; //crossover jaksojen m??r?
vector[N] y; //mittaustulos
vector<lower=0>[N] s; //hoitovasteiden erojen keskivirhe potilaan sis?ll?
real<lower=0> S; //hoitovasteiden y:iden keskihajonta potilaiden v?lill?
}
parameters {
real mu; //populaation keskiarvo
real<lower=0> tau; //populaation keskihajonta
vector[N] theta; //potilastason erot hoitovasteessa
real<lower=0> ultVar; //tuntematon potilaan sis?isen keskivirheen komponentti

real pop_kulma_kesk;
real pop_kulma_var;
real pop_hoitovaikutus_kesk;
real<lower=0> pop_hoitovaikutus_var;
real<lower=0> pop_blokkiefekti;
real<lower=0> pop_hoitojaksoefekti;
real<lower=0> pop_mittausvirhe;
real pot_kulma;
real pot_hoitovaikutus;
real<lower=0> pot_blokkiefekti;
real<lower=0> pot_hoitojaksoefekti;
real<lower=0> pot_mittausvirhe;

}

model {

pop_kulma_kesk ~ normal(0, jotain)
pop_kulma_var ~ cauchy(0, 5);
pop_hoitovaikutus_kesk ~ normal(0, jotain)
pop_hoitovaikutus_var ~ cauchy(0, 5);
pop_blokkiefekti ~ cauchy(0, 5);
pop_hoitojaksoefekti ~ cauchy(0, 5);
pop_mittausvirhe ~ cauchy(0, 5);

kulma ~ normal(pop_kulma_kesk, pop_kulma_var)
hoitovaikutus ~ normal(pop_hoitovaikutus_kesk, pop_hoitovaikutus_var)
blokkiefekti ~ normal(0, pop_blokkiefekti)
hoitojaksoefekti ~ normal(0, pop_hoitojaksoefekti)
mittausvirhe ~ normal(0, pop_mittausvirhe)
y ~ kulma + hoitovaikutus + blokkiefekti + hoitojaksoefekti + mittausvirhe

}

generated quantities {

vector[N] y_rep;
for(n in 1:N)
y_rep[n] = normal_rng(theta[n], sigma[n]);

}
"


#prosessoidaan malli
stanDso <- stan_model(model_code = nOf1)

# Sy?tet??n data mallille ja lasketaan
dataList <- list(N = potilaiden_maara, J = blokkien_maara, y = data1$erojen_keskiarvo, s = data1$erojen_keskivirhe, S = sd(data1$erojen_keskiarvo))
stanFit <- sampling(object = stanDso, data = dataList, chains = 3, iter = 2000, warmup = 200)

# Luodaan erillinen vectori, jossa todelliset keskiarvot. T?t? k?ytet??n testamaan sit?, voiko malli generoida havaittua dataa muistuttavaa aineistoa
y <- data$erojen_keskiarvo

# Tarkastellaan mallin ominaisuuksia Shinystanin verkkosivun kautta
my_shinystan <- as.shinystan(stanFit)
launch_shinystan(my_shinystan)
summary(data1)
