FROM debian:latest

RUN apt-get update && apt-get install -y \
    g++ \
    make \
    build-essential \
    valgrind

WORKDIR /app

COPY . .

RUN make clean

RUN make

CMD ["valgrind", "--tool=cachegrind", "./NN-ab-ovo"]