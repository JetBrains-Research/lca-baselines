 Here is a Python code snippet that uses the Kubernetes Python client to create a deployment, service, and ingress in a Kubernetes cluster. The code assumes that you have already installed the `kubernetes` library.

```python
from kubernetes import client, config

def create_deployment(api_instance, namespace, deployment_name, image, ports):
    container = client.V1Container(
        name=deployment_name,
        image=image,
        ports=ports
    )

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": deployment_name}),
        spec=client.V1PodSpec(containers=[container])
    )

    spec = client.V1DeploymentSpec(
        replicas=3,
        template=template
    )

    body = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=deployment_name),
        spec=spec
    )

    return api_instance.create_namespaced_deployment(namespace, body)

def create_service(api_instance, namespace, deployment_name, selector, type_="ClusterIP", ports=[{"port": 80}]):
    spec = client.V1ServiceSpec(
        selector=selector,
        ports=ports,
        type=type_
    )

    body = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=deployment_name),
        spec=spec
    )

    return api_instance.create_namespaced_service(namespace, body)

def create_ingress(api_instance, namespace, deployment_name, host, path, service_name):
    rules = [client.V1IngressRule(host=host, http=client.V1HTTPIngressRuleValue(paths=[client.V1HTTPIngressPath(path_type="Prefix", path=path, backend=client.V1IngressBackend(service=client.V1ObjectReference(name=service_name, kind="Service")))]))]

    body = client.V1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(name=deployment_name),
        spec=client.V1IngressSpec(rules=rules)
    )

    return api_instance.create_namespaced_ingress(namespace, body)

def main():
    config.load_kube_config()
    core_v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
    extensions_v1beta1 = client.ExtensionsV1beta1Api()

    namespace = "default"
    deployment_name = "my-deployment"
    image = "nginx:latest"
    selector = {"app": deployment_name}

    deployment = create_deployment(apps_v1, namespace, deployment_name, image, [container_port(80)])
    service = create_service(core_v1, namespace, deployment_name, selector)
    ingress = create_ingress(extensions_v1beta1, namespace, deployment_name, "my-host.com", "/path", service.metadata.name)

if __name__ == "__main__":
    main()
```

This code creates a deployment named `my-deployment` using the `nginx:latest` image, a service associated with the deployment, and an ingress that allows external network access to the service on `my-host.com` at the path `/path`. The deployment runs three replicas of the container and exposes port 80.