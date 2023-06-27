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
```

### Installing

- Install Python3.10
- (Optional) Install Pycharm
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
- Install requirements
    - Pycharm IDE
        ```
        Navigate to requirements.txt and follow pop-up 
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

## Running the tests (WIP)
### Integration Testing
```
Current branch: 'feat/communication'
Run tests from postman
```

### Break down into end to end tests (WIP)
```
WIP
```

### And coding style tests (WIP)
```
WIP
```

## Deployment Docker
    - Prerequisites
        ```
        Docker
        ```
    - Clone current branch: 'dev'
    - Create the folder "neurodesktop-storage" in the vm path '/home/ntua'
    - Create the folder "static" in the vm path in '/home/ntua/neurodesktop-storage'
    - Start docker image of Neurodesk using the production folder docker compose 
    - Start docker of analytics backend using the production folder docker compose 
        ```
        docker compose up -d --build 
        ```
## Deployment (Kubernetes) WIP
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

## Built With
WIP 

[//]: # ()
[//]: # (* [SpringBoot]&#40;http://springboot.io&#41; - The Java framework used)

[//]: # (* [Maven]&#40;https://maven.apache.org/&#41; - Dependency Management)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](tags). 

## Authors

* **George Doukas** - *Role* - [gd180](https://github.com/gd180)
* **Loukas Ilias** - *Role* - [loukasilias](https://github.com/loukasilias)
* **Michael Kontoulis** - *Role* - [Mikailo](https://github.com/Mikailo)
* **Christodoulos Santorinaios** - *Role* - [csantor](https://github.com/csantor)

See also the list of [contributors](contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
