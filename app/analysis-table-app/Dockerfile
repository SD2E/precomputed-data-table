FROM sd2e/base:ubuntu18

RUN apt-get update && \
    apt-get install -y \
        build-essential \
		python3 \
		python3-dev \
		python3-pip && \	
    apt-get clean
    
RUN pip3 install --upgrade pip

ADD src /src
ADD config.yml /config.yml