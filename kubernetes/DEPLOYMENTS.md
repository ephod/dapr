# Kubernetes deployments

## Ping-Pong

Apply Kubernetes manifests.

```shell
kubectl apply -f ./ping_pong/service-deployment.yml
kubectl apply -f ./ping_pong/client-deployment.yml
```

Logs.

```shell
# Service
kubectl logs --follow -l app="pingpongactor" -c pingpongactor -n test
kubectl logs --follow -l app="pingpongactor" -n test --all-containers
# Client
kubectl logs --follow -l app="pingpongactor-client" -c pingpongactor-client -n test
kubectl logs --follow -l app="pingpongactor-client" -n test --all-containers
```

Delete Kubernetes manifests.

```shell
kubectl delete -f ./ping_pong/service-deployment.yml
kubectl delete -f ./ping_pong/client-deployment.yml
```

## Bank

Apply Kubernetes manifests.

```shell
kubectl apply -f ./bank/service-deployment.yml
kubectl apply -f ./bank/client-deployment.yml
```

Logs.

```shell
# Service
kubectl logs --follow -l app="bankactor" -c bankactor -n test
kubectl logs --follow -l app="bankactor" -n test --all-containers
# Client
kubectl logs --follow -l app="bankactor-client" -c bankactor-client -n test
kubectl logs --follow -l app="bankactor-client" -n test --all-containers
```

Delete Kubernetes manifests.

```shell
kubectl delete -f ./bank/service-deployment.yml
kubectl delete -f ./bank/client-deployment.yml
```

## MNIST

Apply Kubernetes manifests.

```shell
kubectl apply -f ./mnist/service-deployment.yml
kubectl apply -f ./mnist/client-deployment.yml
```

Logs.

```shell
# Service
kubectl logs --follow -l app="mnistactor" -c mnistactor -n test
kubectl logs --follow -l app="mnistactor" -n test --all-containers
# Client
kubectl logs --follow -l app="mnistactor-client" -c mnistactor-client -n test
kubectl logs --follow -l app="mnistactor-client" -n test --all-containers
```

Delete Kubernetes manifests.

```shell
kubectl delete -f ./mnist/service-deployment.yml
kubectl delete -f ./mnist/client-deployment.yml
```

## Viewing Tracing Data

https://docs.dapr.io/operations/monitoring/tracing/supported-tracing-backends/zipkin/

```shell
kubectl port-forward svc/zipkin 9411:9411
```

## Modify Dapr Kubernete's manifests

https://docs.dapr.io/operations/configuration/increase-request-size/
