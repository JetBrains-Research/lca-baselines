 ```python
from kubernetes import client, config

def create_deployment():
    # Configuration
    config.load_kube_config()

    # Create Deployment object
    deployment = client.AppsV1Api()

    container = client.V1Container(
        name="my-container",
        image="my-image:latest",
        ports=[client.V1ContainerPort(container_port=80)])

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "my-app"}),
        spec=client.V1PodSpec(containers=[container]))

    spec = client.V1DeploymentSpec(
        replicas=3,
        template=template)

    deployment_object = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name="my-deployment"),
        spec=spec)

    # Create Deployment
    deployment.create_namespaced_deployment(body=deployment_object, namespace="default")

def create_service():
    # Configuration
    config.load_kube_config()

    # Create Service object
    service = client.CoreV1Api()

    spec = client.V1ServiceSpec(
        selector={"app": "my-app"},
        ports=[client.V1ServicePort(port=80, target_port=int(80))],
        type="ClusterIP")

    service_object = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name="my-service"),
        spec=spec)

    # Create Service
    service.create_namespaced_service(body=service_object, namespace="default")

def create_ingress():
    # Configuration
    config.load_kube_config()

    # Create Ingress object
    ingress = client.NetworkingV1Api()

    rule = client.NetworkingV1IngressRule(
        host="my-host.example.com",
        http=client.NetworkingV1HTTPIngressRuleValue(
            paths=[client.NetworkingV1HTTPIngressPath(
                path="/path",
                path_type="Prefix",
                backend=client.NetworkingV1IngressBackend(
                    service=client.NetworkingV1IngressServiceBackend(
                        name="my-service",
                        port=client.NetworkingV1ServiceBackendPort(number=80))))]))

    ingress_object = client.NetworkingV1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(name="my-ingress"),
        spec=client.NetworkingV1IngressSpec(rules=[rule]))

    # Create Ingress
    ingress.create_namespaced_ingress(body=ingress_object, namespace="default")

def main():
    create_deployment()
    create_service()
    create_ingress()

if __name__ == '__main__':
    main()
```