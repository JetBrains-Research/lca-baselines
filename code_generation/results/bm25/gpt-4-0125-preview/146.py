```python
from kubernetes import client, config
from kubernetes.client.rest import ApiException

def create_deployment(api_instance, namespace, deployment_name, image, container_port):
    # Define container and deployment specs
    container = client.V1Container(
        name=deployment_name,
        image=image,
        ports=[client.V1ContainerPort(container_port=container_port)]
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": deployment_name}),
        spec=client.V1PodSpec(containers=[container])
    )
    spec = client.V1DeploymentSpec(
        replicas=1,
        template=template,
        selector={'matchLabels': {'app': deployment_name}}
    )
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=deployment_name),
        spec=spec
    )

    # Create deployment
    try:
        api_response = api_instance.create_namespaced_deployment(
            body=deployment,
            namespace=namespace
        )
        print(f"Deployment created. status='{api_response.status}'")
    except ApiException as e:
        print(f"Exception when calling AppsV1Api->create_namespaced_deployment: {e}")

def create_service(api_instance, namespace, service_name, deployment_name, port):
    # Define service spec
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=service_name),
        spec=client.V1ServiceSpec(
            selector={"app": deployment_name},
            ports=[client.V1ServicePort(port=port, target_port=port)]
        )
    )

    # Create service
    try:
        api_response = api_instance.create_namespaced_service(
            namespace=namespace,
            body=service
        )
        print(f"Service created. status='{api_response.status}'")
    except ApiException as e:
        print(f"Exception when calling CoreV1Api->create_namespaced_service: {e}")

def create_ingress(api_instance, namespace, ingress_name, service_name, host, path, port):
    # Define ingress spec
    backend = client.V1IngressBackend(
        service=client.V1IngressServiceBackend(
            name=service_name,
            port=client.V1ServiceBackendPort(number=port)
        )
    )
    path_type = "ImplementationSpecific"
    ingress_path = client.V1HTTPIngressPath(
        path=path,
        path_type=path_type,
        backend=backend
    )
    rule = client.V1IngressRule(
        host=host,
        http=client.V1HTTPIngressRuleValue(paths=[ingress_path])
    )
    spec = client.V1IngressSpec(
        rules=[rule]
    )
    ingress = client.V1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(name=ingress_name),
        spec=spec
    )

    # Create ingress
    try:
        api_response = api_instance.create_namespaced_ingress(
            namespace=namespace,
            body=ingress
        )
        print(f"Ingress created. status='{api_response.status}'")
    except ApiException as e:
        print(f"Exception when calling NetworkingV1Api->create_namespaced_ingress: {e}")

def main():
    config.load_kube_config()
    namespace = "default"
    deployment_name = "example-deployment"
    service_name = "example-service"
    ingress_name = "example-ingress"
    image = "nginx:1.14.2"
    port = 80
    host = "example.com"
    path = "/"

    apps_v1_api = client.AppsV1Api()
    core_v1_api = client.CoreV1Api()
    networking_v1_api = client.NetworkingV1Api()

    create_deployment(apps_v1_api, namespace, deployment_name, image, port)
    create_service(core_v1_api, namespace, service_name, deployment_name, port)
    create_ingress(networking_v1_api, namespace, ingress_name, service_name, host, path, port)

if __name__ == "__main__":
    main()
```