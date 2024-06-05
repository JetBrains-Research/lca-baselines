 import hpeOneView
from hpeOneView.oneview_client import OneViewClient

config_file = "config.json"
oneview_client = OneViewClient.from_json_file(config_file)

scope = oneview_client.scopes.create(name="new_scope")

user_params = {
    "name": "new_user",
    "password": "new_password",
    "scope": scope.name,
    "role": "READ_ONLY",
    "type": "Local",
    "loginId": "new_user@example.com"
}
user = oneview_client.users.create(user_params)

multiple_users = [
    {
        "name": "user1",
        "password": "password1",
        "scope": scope.name,
        "role": "READ_ONLY",
        "type": "Local",
        "loginId": "user1@example.com"
    },
    {
        "name": "user2",
        "password": "password2",
        "scope": scope.name,
        "role": "READ_WRITE",
        "type": "Local",
        "loginId": "user2@example.com"
    }
]
created_users = oneview_client.users.create_multiple(multiple_users)

user.set_password("new_password")

role_to_add = oneview_client.roles.get_by_name("OPERATOR")
user.add_role(role_to_add)

user.update_roles(add_roles=["OPERATOR"], remove_roles=["READ_ONLY"])

user.remove_role("OPERATOR")

get_user = oneview_client.users.get_by_username("new_user")

all_users = oneview_client.users.get_all()

oneview_client.users.validate_username("new_user")
oneview_client.users.validate_full_name("New User")

roles_of_user = get_user.get_roles()

users_by_role = oneview_client.users.get_by_role("READ_ONLY")

oneview_client.users.delete(get_user)

to_delete = [created_users[0], created_users[1]]
oneview_client.users.delete_multiple(to_delete)