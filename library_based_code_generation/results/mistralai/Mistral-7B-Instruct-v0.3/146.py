 Here's a Python script that uses the Kubernetes Python client to create a deployment, service, and ingress in a Kubernetes cluster. The script assumes you have already installed the `kubernetes` and `kubernetes-client` packages.

```python
from kubernetes import client, config

def create_deployment(api_instance):
    v1 = client.AppsV1Api(api_instance)
    container = client.V1Container(
        name="my-container",
        image="my-image:latest",
        ports=[client.V1ContainerPort(container_port=80)]
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "my-app"}),
        spec=client.V1PodSpec(containers=[container])
    )
    spec = client.V1DeploymentSpec(
        replicas=3,
        selector={"app": "my-app"},
        template=template
    )
    body = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name="my-deployment"),
        spec=spec
    )
    return v1.create_namespaced_deployment(body, "default")

def create_service(api_instance):
    v1 = client.CoreV1Api(api_instance)
    spec = client.V1ServiceSpec(
        selector={"app": "my-app"},
        ports=[client.V1ServicePort(port=80, target_port=80)]
    )
    body = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name="my-service"),
        spec=spec
    )
    return v1.create_namespaced_service(body, "default")

def create_ingress(api_instance):
    v1 = client.ExtensionsV1beta1Api(api_instance)
    rule = [client.V1beta1IngressRule(
        host="my-host",
        http=client.V1beta1HTTPIngressRuleValue(
            paths=[client.V1beta1HTTPIngressPath(
                path="/*",
                backend=client.V1beta1IngressBackend(
                    service_name="my-service",
                    service_port=80
                )
            )]
        )
    )]
    spec = client.V1beta1IngressSpec(rules=rule)
    body = client.V1beta1Ingress(
        api_version="extensions/v1beta1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(name="my-ingress"),
        spec=spec
    )
    return v1.create_namespaced_ingress(body, "default")

def main():
    config.load_kube_config()
    api_instance = client.ApiClient()

    deployment = create_deployment(api_instance)
    service = create_service(api_instance)
    ingress = create_ingress(api_instance)

    print(f"Deployment created: {deployment.metadata.name}")
    print(f"Service created: {service.metadata.name}")
    print(f"Ingress created: {ingress.metadata.name}")

if __name__ == "__main__":
    main()
```

Replace `my-image:latest`, `my-host`, and `my-deployment`, `my-service`, `my-ingress` with your desired container image, host, and names for the deployment, service, and ingress respectively.