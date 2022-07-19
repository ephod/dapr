# Dapr Python

## OCI instructions

```shell
docker login docker-registry.docker-registry.svc.cluster.local:5000
docker logout docker-registry.docker-registry.svc.cluster.local:5000
```

```shell
docker build --platform=linux/amd64 -t docker-registry.docker-registry.svc.cluster.local:5000/demo_actor:latest . 
docker image push docker-registry.docker-registry.svc.cluster.local:5000/demo_actor:latest
```
