FROM rocker/verse:3.5.3

# Install ed, since nloptr needs it to compile
# Install ccache to speed up Stan installation
# Install libxt-dev for Cairo
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    clang \
    apt-utils \
    ed \
    ccache \
    libxt-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/

# Set up environment
# Use correct Stan Makevars: https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Mac-or-Linux#prerequisite--c-toolchain-and-configuration
RUN mkdir -p home/.R \
    # Add global configuration files
    && echo "\nCXX=clang++ -ftemplate-depth-256\n" >> home/.R/Makevars \
    && echo "CC=clang\n" >> home/.R/Makevars \
    && echo '\n \
    \n# Stan stuff \
    \nCXXFLAGS+=-O3 -mtune=native -march=native -Wno-unused-variable -Wno-unused-function -Wno-macro-redefined \
    \n' >> home/.R/Makevars \
    # Make R use ccache correctly: http://dirk.eddelbuettel.com/blog/2017/11/27/
    && mkdir -p home/.ccache/ \
    && echo "max_size = 5.0G \
    \nsloppiness = include_file_ctime \
    \nhash_dir = false \
    \n" >> home/.ccache/ccache.conf \
    # Add configuration files for RStudio user
    && mkdir -p /home/rstudio/.R/ \
    && echo '\n \
    \n# Stan stuff \
    \nCXXFLAGS=-O3 -mtune=native -march=native -Wno-unused-variable -Wno-unused-function -Wno-macro-redefined \
    \n' >> /home/rstudio/.R/Makevars \
    && echo "rstan::rstan_options(auto_write = TRUE)\n" >> /home/rstudio/.Rprofile \
    && echo "options(mc.cores = parallel::detectCores())\n" >> /home/rstudio/.Rprofile

# Install Stan, rstan, rstanarm, brms, and friends
RUN install2.r --error --deps TRUE \
    rstan \
    bayesplot \
    rstanarm \
    # rstantools \
    shinystan \
    && rm -rf /tmp/downloaded_packages/ /tmp/*.rds

# Install other packages for timeline analysis
RUN install2.r --error --deps TRUE \
    # need to install XLConnect explicitly before others to avoid dependency issues
    XLConnect \
    zoo \
    xts \
    tidyquant \
    && rm -rf /tmp/downloaded_packages/ /tmp/*.rds
