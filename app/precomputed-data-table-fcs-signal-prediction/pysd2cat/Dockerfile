FROM sd2e/apps:python3

RUN pip install FlowCytometryTools

ADD . /pysd2cat
RUN cd /pysd2cat && \
    python3 setup.py install
