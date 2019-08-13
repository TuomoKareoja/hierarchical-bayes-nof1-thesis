# MALLIN SPESIFIOINTI 

#T‰ss‰ mallissa jokaista blokkia k‰sitell‰‰n kolikonheittona

#ladataan tarvittavat paketit

library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
library(shinystan)
library(dplyr)

#muokataan data STANille sopivaan muotoon

data2 <- data %>% 
  group_by(Potilas, id_blokki, id_jakso, hoito_vai_plasebo) %>% 
  summarise(keskiarvo_jakso = mean(Tulos))

data2 <- data2 %>% 
  arrange(Potilas, id_blokki, hoito_vai_plasebo) %>% 
  group_by(Potilas, id_blokki) %>% 
  mutate(plasebo = lag(keskiarvo_jakso)) %>%
  filter(hoito_vai_plasebo==1) %>%
  transmute(ero = keskiarvo_jakso - plasebo) %>%
  transmute(hoito_plaseboa_parempi = ifelse(ero > 0, 1, 0)) %>%
  ungroup() %>%
  group_by(Potilas) %>%
  summarise(hoito_parempi_maara = sum(hoito_plaseboa_parempi))



#Mallien spesifiointi


# malli jossa hajonta osittain tunnettu
nOf1 = "
data {
int<lower=0> N; //potilaiden m‰‰r‰
int<lower=2> B; //blokkien m‰‰r‰
int<lower=0> y[N]; //blokkien m‰‰r‰, jossa hoito plaseboa parempi
}
parameters {
real mu; //populaation keskiarvo
real<lower=0.001> ss; //populaation keskihajonta. M‰‰ritetty niin, ett‰ keskihajonta ei voi olla 0
real<lower=0, upper=1> p[N]; //potilastason todenn‰kˆisyys siihen, ett‰ aito hoito parempi
}
model {
mu ~ uniform(0, 1); // populaation jakauman keskiarvon priori asetetaan tasajakaumaksi
ss ~ gamma(0.01, 0.01); // populaation hajonnan prioriksi asetetaan gammajakauma. T‰m‰ voi olla ongemallinen laskennallisista syist‰ johtuen 
p ~ beta(mu*ss, (1-mu)*ss); //huomaa parametrien hierarkisuus!
y ~ binomial(B, p);
}
generated quantities {
vector[N] y_rep;
for(n in 1:N)
y_rep[n] = binomial_rng(B, p[n]);
}
"


#prosessoidaan malli
stanDso <- stan_model(model_code = nOf1)

# Syˆtet‰‰n data mallille ja lasketaan
dataList <- list(N = potilaiden_maara, B = blokkien_maara, y = data2$hoito_parempi_maara)
stanFit <- sampling(object = stanDso, data = dataList, chains = 3, iter = 2000, warmup = 200)

# Luodaan erillinen vectori, jossa todelliset keskiarvot. T‰t‰ k‰ytet‰‰n testamaan sit‰, voiko malli generoida havaittua dataa muistuttavaa aineistoa
y <- data$erojen_keskiarvo

# Tarkastellaan mallin ominaisuuksia Shinystanin verkkosivun kautta
my_shinystan <- as.shinystan(stanFit)
launch_shinystan(my_shinystan)
summary(data2)
