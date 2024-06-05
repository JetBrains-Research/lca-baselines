```python
from kubernetes import client, config
from kubernetes.client import V1Deployment, V1Service, ExtensionsV1beta1Ingress
from kubernetes.client.rest import ApiException

def create_deployment(api_instance, image, name="my-deployment", replicas=1, container_port=80):
    container = client.V1Container(
        name=name,
        image=image,
        ports=[client.V1ContainerPort(container_port=container_port)]
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": name}),
        spec=client.V1PodSpec(containers=[container])
    )
    spec = client.V1DeploymentSpec(
        replicas=replicas,
        template=template,
        selector={'matchLabels': {'app': name}}
    )
    deployment = V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=name),
        spec=spec
    )

    try:
        api_response = api_instance.create_namespaced_deployment(
            body=deployment,
            namespace="default"
        )
        print("Deployment created. status='%s'" % str(api_response.status))
    except ApiException as e:
        print("Exception when calling AppsV1Api->create_namespaced_deployment: %s\n" % e)

def create_service(api_instance, name="my-service", port=80, target_port=80):
    body = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name=name
        ),
        spec=client.V1ServiceSpec(
            selector={"app": name},
            ports=[client.V1ServicePort(port=port, target_port=target_port)],
            type="NodePort"
        )
    )

    try:
        api_response = api_instance.create_namespaced_service(
            namespace="default",
            body=body
        )
        print("Service created. status='%s'" % str(api_response.status))
    except ApiException as e:
        print("Exception when calling CoreV1Api->create_namespaced_service: %s\n" % e)

def create_ingress(api_instance, name="my-ingress", service_name="my-service", host="example.com", path="/", service_port=80):
    backend = client.V1beta1IngressBackend(service_name=service_name, service_port=service_port)
    path = client.V1beta1HTTPIngressPath(backend=backend, path=path)
    rule = client.V1beta1IngressRule(host=host, http=client.V1beta1HTTPIngressRuleValue(paths=[path]))
    spec = client.ExtensionsV1beta1IngressSpec(rules=[rule])
    ingress = ExtensionsV1beta1Ingress(
        api_version="extensions/v1beta1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(name=name),
        spec=spec
    )

    try:
        api_response = api_instance.create_namespaced_ingress(
            namespace="default",
            body=ingress
        )
        print("Ingress created. status='%s'" % str(api_response.status))
    except ApiException as e:
        print("Exception when calling ExtensionsV1beta1Api->create_namespaced_ingress: %s\n" % e)

def main():
    config.load_kube_config()
    apps_v1_api = client.AppsV1Api()
    core_v1_api = client.CoreV1Api()
    extensions_v1beta1_api = client.ExtensionsV1beta1Api()

    image = "nginx:latest"
    deployment_name = "nginx-deployment"
    service_name = "nginx-service"
    ingress_name = "nginx-ingress"
    port = 80
    host = "nginx.example.com"
    path = "/"

    create_deployment(apps_v1_api, image=image, name=deployment_name, container_port=port)
    create_service(core_v1_api, name=service_name, port=port, target_port=port)
    create_ingress(extensions_v1beta1_api, name=ingress_name, service_name=service_name, host=host, path=path, service_port=port)

if __name__ == "__main__":
    main()
```