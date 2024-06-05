 ```python
from kubernetes import client, config

def create_deployment():
    config.load_kube_config()
    api_instance = client.AppsV1Api()
    container = client.V1Container(
        name="example-container",
        image="example-image",
        ports=[client.V1ContainerPort(container_port=80)])
    spec = client.V1DeploymentSpec(
        replicas=1,
        selector={'matchLabels': {'app': 'example'}},
        template=client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={'app': 'example'}),
            spec=client.V1PodSpec(containers=[container])))
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name="example-deployment"),
        spec=spec)
    try:
        api_instance.create_namespaced_deployment(namespace="default", body=deployment)
    except ApiException as e:
        print("Exception when calling AppsV1Api->create_namespaced_deployment: %s\n" % e)

def create_service():
    config.load_kube_config()
    api_instance = client.CoreV1Api()
    spec = client.V1ServiceSpec(
        selector={'app': 'example'},
        ports=[client.V1ServicePort(port=80, target_port=int32_t(80))])
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name="example-service"),
        spec=spec)
    try:
        api_instance.create_namespaced_service(namespace="default", body=service)
    except ApiException as e:
        print("Exception when calling CoreV1Api->create_namespaced_service: %s\n" % e)

def create_ingress():
    config.load_kube_config()
    api_instance = client.NetworkingV1Api()
    rules = [client.V1IngressRule(
        host="example.com",
        http=client.V1HTTPIngressRuleValue(
            paths=[client.V1HTTPIngressPath(
                path="/example",
                path_type="Prefix",
                backend=client.V1IngressBackend(
                    service=client.V1IngressServiceBackend(
                        name="example-service",
                        port=client.V1ServiceBackendPort(number=80))))]))]
    ingress = client.V1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(name="example-ingress"),
        spec=client.V1IngressSpec(rules=rules))
    try:
        api_instance.create_namespaced_ingress(namespace="default", body=ingress)
    except ApiException as e:
        print("Exception when calling NetworkingV1Api->create_namespaced_ingress: %s\n" % e)

def main():
    create_deployment()
    create_service()
    create_ingress()

if __name__ == '__main__':
    main()
```