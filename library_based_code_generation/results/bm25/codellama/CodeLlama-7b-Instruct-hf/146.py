  ```
from kubernetes import client, config

def create_deployment(name, image, port):
    deployment = client.V1Deployment(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1DeploymentSpec(
            replicas=1,
            selector=client.V1LabelSelector(match_labels={"app": name}),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels={"app": name}),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name=name,
                            image=image,
                            ports=[client.V1ContainerPort(container_port=port)]
                        )
                    ]
                )
            )
        )
    )
    return deployment

def create_service(name, port):
    service = client.V1Service(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1ServiceSpec(
            selector={"app": name},
            ports=[client.V1ServicePort(port=port)]
        )
    )
    return service

def create_ingress(name, host, path):
    ingress = client.V1Ingress(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1IngressSpec(
            rules=[
                client.V1IngressRule(
                    host=host,
                    http=client.V1HTTPIngressRuleValue(
                        paths=[
                            client.V1HTTPIngressPath(
                                path=path,
                                backend=client.V1IngressBackend(service_name=name, service_port=port)
                            )
                        ]
                    )
                )
            ]
        )
    )
    return ingress

def main():
    config.load_kube_config()
    deployment = create_deployment("my-deployment", "my-image", 8080)
    service = create_service("my-service", 8080)
    ingress = create_ingress("my-ingress", "my-host", "/my-path")
    client.CoreV1Api().create_namespaced_deployment("default", deployment)
    client.CoreV1Api().create_namespaced_service("default", service)
    client.NetworkingV1beta1Api().create_namespaced_ingress("default", ingress)

if __name__ == "__main__":
    main()
```