from hpeOneView.oneview_client import OneViewClient

config = {
    "ip": "192.168.1.10",
    "credentials": {
        "userName": "admin",
        "password": "password"
    }
}

oneview_client = OneViewClient(config)

scope = oneview_client.create_scope({"name": "Scope1"})

user1 = oneview_client.create_user({"userName": "user1", "password": "password1", "permissions": ["read"]})
user2 = oneview_client.create_user({"userName": "user2", "password": "password2", "permissions": ["write"]})

oneview_client.update_user_password("user1", "newpassword")

oneview_client.add_role_to_userName("user1", "newrole")

oneview_client.update_role_to_userName("user1", ["read", "write"])

oneview_client.remove_role_to_userName("user1", "write")

user = oneview_client.get_user_by_username("user1")

users = oneview_client.get_all_users()

is_fullname_in_use = oneview_client.validate_fullname("John Doe")

is_username_in_use = oneview_client.validate_username("johndoe")

roles = oneview_client.get_roles_for_user("user1")

users_by_role = oneview_client.get_users_by_role("admin")

oneview_client.delete_user("user1")

oneview_client.delete_multiple_users(["user1", "user2"])