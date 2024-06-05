 ```python
from hpeOneView.oneview_client import OneViewClient
from getpass import getpass


def load_configuration(config_file):
    with open(config_file) as f:
        config = [x.strip().split("=") for x in f.readlines()]
    return dict(x for x in config)


def connect_to_client(config):
    oneview_client = OneViewClient(config['host'],
                                   config['user'],
                                   config['password'],
                                   api_version=config['api_version'])
    oneview_client.login()
    return oneview_client


def create_scope(oneview_client, scope_name):
    scope = oneview_client.scopes.create(name=scope_name)
    return scope


def create_user(oneview_client, username, password, scope_uri, role_uris):
    user = oneview_client.users.create(username=username,
                                       password=password,
                                