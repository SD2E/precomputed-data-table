FROM sd2e/reactors:python3-edge

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade git+https://github.com/SD2E/python-datacatalog.git@2_2
RUN pip3 install --upgrade git+https://github.com/SD2E/bacanora.git@master

COPY record_product_info.py /record_product_info.py
COPY external_apps /external_apps
COPY version.txt /version.txt