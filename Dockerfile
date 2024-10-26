FROM python:3.6

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update -y \
    && apt-get upgrade -y --allow-unauthenticated \
    && apt-get install -y --no-install-recommends \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir \
        tensorflow==1.10.0 \
        tqdm \
        numpy \
        pandas \
        rdkit-pypi \
        scikit-learn \
        matplotlib \
        dwave-system \
        dwave-neal \
    && pip3 uninstall -y \
        numpy \
        dwave-system \
    && pip3 install --no-cache-dir \
        numpy \
        dwave-system \
        dwave-neal

WORKDIR /qa_mid
CMD ["/bin/bash"]
