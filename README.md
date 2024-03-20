# MES-CoBraD Analytics and Visualisation Backend Module
![REPO-TYPE](https://img.shields.io/badge/repo--type-backend-critical?style=for-the-badge&logo=github)

This is the backend of the MES-CoBraD Data Analytics and Visualisation Module

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
```
Python 3.10 https://www.python.org/downloads/
```
#### Optional but recommended
```
Pycharm IDE
Docker
```

### Installing

- Install Python3.10
- (Optional) Install Pycharm
- (Optional) Install latest version of docker (current ver.4.26.1)
- Clone this repository
  - Pycharm IDE 
    ```
    - Get from VCS
    - Github
    - Select this repository
    ```
  - cmd
    ```
    git clone https://github.com/mescobrad-project/data-analytics-and-visualisation-backend.git
    ```
- Create VENV 
    - Pycharm IDE
    -   ```
        Follow pop-up 
        ```
    - cmd
    -   ```
        python3 -m venv <myenvname>
        ```
- Install requirements
    - Pycharm IDE
        ```
        Navigate to requirements.txt and follow pop-up 
        ```
    - cmd
    -   ```
        pip install -r /path/to/requirements.txt
        ```
### Running 
    - Pycharm IDE
        ```
           Edit configuraiton -> main.py
        ```
     - CMD
        ```
            uvicorn main:app --reload
        ```
    - Docker 
        ```
            Follow docker deployement instructions below
        ```
## Testing
### Integration Testing 
Current branch: 'feat/unit_test' and 'dev'
Testing running the following commands:
Run pytest to run all testing files:
(Might result to errors depending on final implementation of tests. Highly recommend to run files on their own)
```
pytest
```

To disable all warning run
```
pytest  --disable-warnings
```

To run a specific testing file 
```
pytest app/test_eeg.py --disable-warnings
```

## Deployment Docker Local
    - Prerequisites
        ```
        Docker
        ```
    - Clone latest branch: 'dev'
    - Create the folder "neurodesktop-storage" in '/' or 'C:/' 
    - Create the folder "static" in the vm path in '/neurodesktop-storage' or 'C:/neurodesktop-storage'
    - Start docker of all modules using the docker compose file found in project location 'data-analytics-and-visualisation-backend/docker/' 
        ```
        docker compose up -d --build 
        ```
## Deployment Docker Simavi Server
    - Prerequisites
        ```
        Docker
        ```
    - Clone latest branch: 'prod_new'
    - Create the folder "neurodesktop-storage" in the vm path '/home/ntua'
    - Create the folder "static" in the vm path in '/neurodesktop-storage' or 'C:/neurodesktop-storage'
    - Start docker of all modules using the docker compose file found in project location 'data-analytics-and-visualisation-backend/docker/docker_production' 
        ```
        docker compose up -d --build 
        ```
<!-- ## Deployment (Kubernetes) WIP -legacy not used
    - Prerequisites
        ```
        Kuberenetes : Recommended enable through docker desktop
        ```
    - kubectl create -f .\analytics-backend-claim0-persistentvolumeclaim.yaml
    - kubectl create -f .\analytics-backend-service.yaml
    - kubectl create -f .\analytics-backend-deployment.yaml
    - kubectl create -f .\analytics-network-networkpolicy.yaml

    - Misc 
        ```
        kubectl get services -o wide
        kubectl get pods -o wide
        kubectl get nodes -o wide
        kubectl describe services
        kubectl describe services/analytics-backend
        
        To stop the nodes:
        kubectl get deployments
        kubectl delete deployment analytics-backend
        

        To clear:  kubectl delete all --all --namespace default 

        To see logs, can be done through docker-desktop
        ```
--> 
## Built With
[FastAPI](https://fastapi.tiangolo.com/)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning
Not currently following any including semver - will try to follow it.
<!--  We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](tags). --> 

## Authors

* **George Doukas** - *Dev* - [gd180](https://github.com/gd180)
* **Michael Kontoulis** - *Dev* - [Mikailo](https://github.com/Mikailo)
* **Loukas Ilias** - *Dev* - [loukasilias](https://github.com/loukasilias)
* **Theodoros Pountisis** - *Dev* - [Mikailo](https://github.com/theopnt)
* **George Ladikos** - *Dev* - [Mikailo](https://github.com/georgeladikos)
* **Christodoulos Santorinaios** - *Former Dev* - [csantor](https://github.com/csantor)

See also the list of [contributors](contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
