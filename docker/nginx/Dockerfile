FROM nginx
RUN apt-get update
COPY nginx.conf /etc/nginx/nginx.conf
COPY ./certs/self_ssl.pem /etc/nginx/certificate/nginx-certificate.crt
COPY ./certs/self_ssl_key.pem /etc/nginx/certificate/nginx.key
COPY ./certs2/fullchain.pem /etc/nginx/certificate/fullchain.pem
COPY ./certs2/privkey.pem /etc/nginx/certificate/privkey.pem
COPY nginx.conf /etc/nginx/nginx.conf
