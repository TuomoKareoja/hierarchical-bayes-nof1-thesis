library(zoo)
library(ggplot2)
library(tidyr)

set.seed(127)


##############
# PARAMETERS #
##############

# STUDY DESIGN
patient_n <- 40
treatment_n <- 2 # including placebo
block_n <- 4
treatment_measurement_n <- 7
total_measurement_n <- block_n * treatment_n * treatment_measurement_n
balanced_design = TRUE
if(block_n %% 2 != 0){
   stop("Cannot do balanced design with uneven number of blocks")
}

# POPULATION TREATMENT EFFECT
population_treatment_effect_mean_array <- c(0, 1.5)
population_treatment_effect_sd_array <- c(0, 0.9)

# POPULATION UNRELATED LINEAR TIMETREND
population_trend_mean <- 0.02
population_trend_sd <- 0.01

# POPULATION MEASUREMENT ERROR (gamma distribution)
population_measurement_error_shape <- 0.02
population_measurement_error_scale <- 0.005

# POPULATION MEASUREMENT AUTOCORRELATION
population_autocorrelation_alpha <- 80
population_autocorrelation_beta <- 200


##################################################
# CALCULATING PARAMETERS FOR INDIVIDUAL PATIENTS #
##################################################

patient_treatment_effect_array <- rnorm(patient_n, mean = population_treatment_effect_mean_array, sd = population_treatment_effect_sd_array)
patient_trend <- rnorm(patient_n, mean = population_trend_mean, sd = population_trend_sd)
patient_measurement_error <- rgamma(patient_n, shape = pop_mittausvirhe_shape, scale = pop_mittausvirhe_scale)
patient_autocorrelation <- rbeta(patient_n, shape1 = pop_autokor_alfa, shape2 = pop_autokor_beta)

#satunnaistetaan potilaiden hoitojen j�rjestys
satunnaista_hoidot <- function(){
  pop_jarjestys <- list()
  for(potilas in 1:potilaiden_maara){
    pot_jarjestys <- c()
    if(balansoitu){
      for(blokki in 1:(blokkien_maara/2)){
        blokin_hoidot <- c(0,0) #molemmat hoidot plaseboja
        hoidon_paikka <- sample(1:2, 1) # arvotaan kumplaan slottiin hoito laitetaan
        blokin_hoidot[hoidon_paikka] <- 1
        pot_jarjestys <- append(pot_jarjestys, append(blokin_hoidot, rev(blokin_hoidot))) # lis�t��n toinen k��nnetty blokki
      }
    } else {
      for(blokki in 1:blokkien_maara){
        blokin_hoidot <- c(0,0) #molemmat hoidot plaseboja
        hoidon_paikka <- sample(1:2, 1) # arvotaan kumplaan slottiin hoito laitetaan
        blokin_hoidot[hoidon_paikka] <- 1
        pot_jarjestys <- append(pot_jarjestys, blokin_hoidot)
      }
    }
    pop_jarjestys[[potilas]] <- pot_jarjestys
  }
  return(pop_jarjestys)
}
pot_hoitojarjestys <- satunnaista_hoidot()


# Potilaan yksitt�isen mittauksen kaava:
# mittaus = hoitoindikaattori * potilaan hoitovaikutus + mittauksen numero * potilaan trendi + mittausvirhe + jotain

# potilaan mittausten aikasarja ilman trendi�. Aikasarjan satunnaisvaihtelun ajatellaan olevan todellista vaihtelua mitattavassa ominaisuudessa
create_data <- function(){
    
  # Luodaan tyhj� matriisi, johon laitetaan tarvittavat tiedot
  data <- matrix(data = NA, nrow = mittausten_maara, ncol = potilaiden_maara)
  
  for (potilas in 1:potilaiden_maara){

    # Luodaan potilaan havaintosarja
    pot_mittaukset <- (
    
      # Luodaan ARIMA-malli
      arima <- (arima.sim(
        model = list(ar = pot_autokor[potilas]),
        # innov = rnorm(mittausten_maara, 0.02),
        # start.innov = rnorm(mittausten_maara, sd = 0.02),
        n = mittausten_maara
      ) +
      # lis�t��n trendi
      pot_trendi[potilas] * seq(1,mittausten_maara) +
      # Lis�t��n mittausvirhe
      rnorm(n = mittausten_maara, mean = 0, sd = pot_mittausvirhe[potilas]) +
      #Lis�t��n hoitovaikutus
      pot_hoitovaikutus[potilas] * rep(pot_hoitojarjestys[[potilas]], each=jakson_mittausten_maara))
    
    )
    
    #Lis�t��n potilaan mittaukset matriisiin
    data[,potilas] <- pot_mittaukset
  }
  
  data <- as.data.frame(data)
  data <- cbind(data, c(1:mittausten_maara))
  colnames(data) <- c(c(1:potilaiden_maara), "index")
  data <- read.zoo(data, index.column = "index")
  return(data)
  
}


data <- create_data()

autoplot(data, facet = NULL)


# Muokataan data helpommin k�sitelt�v��n muotoon, jossa yksi vain yksi havaintokolumni ja muut indeksej�

data <- as.data.frame(data)

data <- gather(data, key = "Potilas", value = 'Tulos')

data$id_blokki <- rep(
    rep(c(1:blokkien_maara), each=jakson_mittausten_maara * 2)
    , potilaiden_maara
  )

data$id_jakso <- rep(
    rep(c(1:2), each=jakson_mittausten_maara)
    , potilaiden_maara * blokkien_maara
  )

data$id_mittaus <- rep(c(1:jakson_mittausten_maara), potilaiden_maara * blokkien_maara * 2)


data$hoito_vai_plasebo <- rep(unlist(pot_hoitojarjestys), each=jakson_mittausten_maara)

data$Potilas <- as.integer(data$Potilas)
pot_autokor
