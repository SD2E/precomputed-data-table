FROM sd2e/python3:miniconda

ARG LANGUAGE=python3-miniconda
ARG VERSION=0.5.5
ARG AGAVEPY_BRANCH=develop

# Discoverable version inside the container
RUN echo "TACC.cloud Apps\nType: ${LANGUAGE}\nVersion: ${VERSION}" > /etc/apps-VERSION

# Helpful env variable
ENV APPS_VERSION=${VERSION}

RUN conda update -n base conda

# Install R
RUN export DEBIAN_FRONTEND=noninteractive \
     && apt-get update \
     && apt install -y dirmngr apt-transport-https software-properties-common gnupg2 \
#     && apt-key adv --keyserver keys.gnupg.net --recv-key E19F5F87128899B192B1A2C2AD5F960A256A04AF \
     && apt-key adv --keyserver keyserver.ubuntu.com --recv-key E19F5F87128899B192B1A2C2AD5F960A256A04AF \
     && add-apt-repository 'deb https://cloud.r-project.org/bin/linux/debian stretch-cran35/' \
     && apt update \
     && apt install -y r-base
    
# Install edgeR
RUN Rscript -e 'install.packages("BiocManager", repos="http://cran.us.r-project.org"); BiocManager::install("edgeR")'

ADD config.yml /config.yml
ADD src /src

COPY xplan-od-growth-analysis /src/xplan-od-growth-analysis
COPY omics_tools /src/omics_tools

ADD analysis_envs /analysis_envs
RUN conda env create -f /analysis_envs/xplan-od-growth-analysis.yml
RUN conda env create -f /analysis_envs/wasserstein_tenfold_comparisons.yml
RUN conda env create -f /analysis_envs/omics_tools.yml