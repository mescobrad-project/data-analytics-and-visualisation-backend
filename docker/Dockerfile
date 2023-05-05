# use python 3.10 image
FROM python:3.10

# set working directory
WORKDIR /data-analytics-and-visualisation-backend

# install Project requirements
COPY ../requirements.txt .
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# copy contents to container automatic-forensic-tool
ADD .. .

#
CMD ["uvicorn", "app.main:app", "--proxy-headers","--host", "0.0.0.0", "--port", "8000"]
#CMD ["uvicorn", "app.main:app", "--proxy-headers","--host", "0.0.0.0", "--port", "8000"]
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "ssl_keyfile", "./nginx/certs/self_ssl.pem","ssl_certfile", "./nginx/certs/self_ssl_key.pem"]
