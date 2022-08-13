<div id="top"></div>

# Dapr and actor model

## About the project

The actor model is a math model for concurrent computations, where the 🧔 actor symbolizes the universal primitive of concurrent computation.

For this project, I will showcase three examples where Dapr (Python SDK) can work using the actor model. This project will run within Kubernetes.

This project is a technical demonstration of what you can achieve with virtual actors and how to adapt our code.

## Getting started

### Prerequisites

Creating a virtual environment, activating it, and installing the necessary packages.

```shell
# Create Python's virtual environment
python3 -m venv .venv
# Activate Python's virtual environment
source .venv/bin/activate
# Upgrade pip to the latest version
python -m pip install --upgrade pip
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation

```shell
# Production
python -m pip install -r requirements.txt
# Development
# python -m pip install -r requirements-dev.txt
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Uninstall

Exit Python's virtual environment.

```shell
# Exit Python's virtual environment
deactivate
# Remove Python's virtual environment folder
rm -rf .venv/
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Create Docker image

Docker will create an image with is compatible with the Open Container Initiative (OCI).

Login to your own private or public docker repository. You can port forward the deployed Docker registry to your host.

```shell
docker login docker-registry.docker-registry.svc.cluster.local:5000
# docker login host:port
```

Build your image based on a `Dockerfile` file, ignore what is not needed within the `.dockerignore` file, tag this build, and based it for x86 platforms if you are running this on a different architecture.

```shell
docker build --platform=linux/amd64 -t docker-registry.docker-registry.svc.cluster.local:5000/demo_actor:latest . 
```

Push this image to your Dokcer repository.

```shell
docker image push docker-registry.docker-registry.svc.cluster.local:5000/demo_actor:latest
```

Logout of Docker registry.

```shell
docker logout docker-registry.docker-registry.svc.cluster.local:5000
```
