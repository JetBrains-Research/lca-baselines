```python
from kubernetes import client, config

def create_deployment(api_instance):
    container = client.V1Container(
        name="my-container",
        image="my-image",
        ports=[client.V1ContainerPort(container_port=80)]
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "my-app"}),
        spec=client.V1PodSpec(containers=[container])
    )
    spec = client.ExtensionsV1beta1DeploymentSpec(replicas=1, template=template)
    deployment = client.ExtensionsV1beta1Deployment(
        metadata=client.V1ObjectMeta(name="my-deployment"),
        spec=spec
    )
    api_instance.create_namespaced_deployment(namespace="default", body=deployment)

def create_service(api_instance):
    service = client.V1Service(
        metadata=client.V1ObjectMeta(name="my-service"),
        spec=client.V1ServiceSpec(selector={"app": "my-app"}, ports=[client.V1ServicePort(port=80)])
    )
    api_instance.create_namespaced_service(namespace="default", body=service)

def create_ingress(api_instance):
    ingress = client.NetworkingV1beta1Ingress(
        metadata=client.V1ObjectMeta(name="my-ingress"),
        spec=client.NetworkingV1beta1IngressSpec(
            rules=[client.NetworkingV1beta1IngressRule(
                host="example.com",
                http=client.NetworkingV1beta1HTTPIngressRuleValue(
                    paths=[client.NetworkingV1beta1HTTPIngressPath(
                        path="/",
                        backend=client.NetworkingV1beta1IngressBackend(service_name="my-service", service_port=80)
                    )]
                )
            )]
        )
    )
    api_instance.create_namespaced_ingress(namespace="default", body=ingress)

def main():
    config.load_kube_config()
    api_instance = client.AppsV1beta1Api()
    create_deployment(api_instance)
    create_service(api_instance)
    create_ingress(api_instance)

if __name__ == "__main__":
    main()
```