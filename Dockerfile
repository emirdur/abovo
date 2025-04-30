FROM debian:latest

RUN apt-get update && apt-get install -y \
    g++ \
    make \
    build-essential

WORKDIR /app

COPY . .

RUN make

CMD ["./NN-ab-ovo"]
