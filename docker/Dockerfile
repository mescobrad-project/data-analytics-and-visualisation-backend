# use python 3.10 image
FROM python:3.10

# set working directory
WORKDIR /data-analytics-and-visualisation-backend
RUN #apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install Project requirements
COPY ../requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir --upgrade numpy==1.22.4
RUN #pip3 install --no-cache-dir --upgrade numpy==1.21.6
RUN pip3 install --no-cache-dir --upgrade scipy==1.7.3
RUN pip3 install --no-cache-dir --upgrade Cython==0.29.32
RUN pip3 install --no-cache-dir --upgrade scikit-learn==1.0.1 --no-build-isolation
RUN pip3 install --no-cache-dir --upgrade semopy==2.3.10
RUN pip3 install --no-cache-dir --upgrade --no-build-isolation -r requirements.txt

# copy contents to container automatic-forensic-tool
ADD .. .

#
CMD ["uvicorn", "app.main:app", "--proxy-headers","--host", "0.0.0.0", "--port", "8000"]
#CMD ["uvicorn", "app.main:app", "--proxy-headers","--host", "0.0.0.0", "--port", "8000"]
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "ssl_keyfile", "./nginx/certs/self_ssl.pem","ssl_certfile", "./nginx/certs/self_ssl_key.pem"]
