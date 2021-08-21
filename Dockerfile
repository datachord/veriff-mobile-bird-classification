# docker build --build-arg SSH_PRIVATE_KEY="$(cat ../id_rsa)" -t vinnee/mobile-bird . && docker rmi -f $(docker images -q --filter label=stage=app)
# docker run -it --mount type=bind,source="$(pwd)",target=/root/external --name mobile-bird vinnee/mobile-bird
# docker push vinnee/mobile-bird


FROM tensorflow/tensorflow:2.5.0rc0 as packages

# install packages
RUN apt-get update && \
    # fix cv2 ImportError: libGL.so.1: cannot open shared object file: No such file or directory
    apt-get install ffmpeg libsm6 libxext6  -y && \
    #
    pip install tensorflow_hub==0.11.0 && \
    pip install opencv-python==4.5.1.48 && \
    pip install pandas==1.1.5 && \
    pip install matplotlib==3.3.4 && \
    pip install ipytest==0.11.0 && \
    pip install pytest==6.2.4 && \
    pip install Flask==2.0.1 && \
    apt-get install -y openssh-client && \
    apt-get install -y git && \
    apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip/* && rm -rf /usr/local/src/*

FROM packages as app
LABEL stage=app

ARG SSH_PRIVATE_KEY

# add credentials – delete stage after build
RUN mkdir /root/.ssh/ && \
    echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_rsa && \
    chmod -R 600 /root/.ssh/ && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    #
    # clone app
    git clone git@github.com:datachord/veriff-mobile-bird-classification.git && \
    apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip/* && rm -rf /usr/local/src/*

FROM packages

# copy app
COPY --from=app /veriff-mobile-bird-classification /root/veriff-mobile-bird-classification
WORKDIR /root/veriff-mobile-bird-classification
CMD ["python", "classifier.py"]