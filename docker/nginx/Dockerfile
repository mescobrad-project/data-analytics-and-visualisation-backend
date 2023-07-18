FROM nginx
RUN apt-get update
COPY nginx.conf /etc/nginx/nginx.conf
COPY ./certs/self_ssl.pem /etc/nginx/certificate/nginx-certificate.crt
COPY ./certs/self_ssl_key.pem /etc/nginx/certificate/nginx.key
COPY nginx.conf /etc/nginx/nginx.conf
