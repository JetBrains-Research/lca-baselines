```python
from hpeOneView.oneview_client import OneViewClient

# Load configuration to connect to OneView
config = {
    "ip": "<HPE_ONEVIEW_IP>",
    "credentials": {
        "userName": "<USERNAME>",
        "password": "<PASSWORD>"
    }
}

# Establish connection
oneview_client = OneViewClient(config)

# Create a scope
scope_options = {
    "name": "NewScope",
    "description": "Sample Scope Description"
}
scope = oneview_client.scopes.create(scope_options)

# Create a user
user_options = {
    "userName": "newUser",
    "password": "password",
    "fullName": "New User",
    "emailAddress": "newuser@example.com",
    "officePhone": "123-456-7890",
    "mobilePhone": "098-765-4321",
    "roles": ["Read only"],
    "scopeUri": scope['uri']
}
user = oneview_client.security.users.create(user_options)

# Create multiple users with different permissions
users_options = [
    {
        "userName": "user1",
        "password": "password1",
        "fullName": "User One",
        "roles": ["Infrastructure administrator"],
    },
    {
        "userName": "user2",
        "password": "password2",
        "fullName": "User Two",
        "roles": ["Read only"],
    }
]
for user_option in users_options:
    oneview_client.security.users.create(user_option)

# Update a user's password
update_password = {"password": "newPassword"}
oneview_client.security.users.update(update_password, "newUser")

# Add a new role to an existing user
user['roles'].append('Network administrator')
oneview_client.security.users.update(user, user['userName'])

# Update the roles of a user
user['roles'] = ['Storage administrator']
oneview_client.security.users.update(user, user['userName'])

# Remove certain roles from a user
user['roles'].remove('Storage administrator')
oneview_client.security.users.update(user, user['userName'])

# Retrieve a user by their username
retrieved_user = oneview_client.security.users.get_by('userName', 'newUser')[0]

# Retrieve all users
all_users = oneview_client.security.users.get_all()

# Validate if a full name or username is already in use
def validate_user_existence(username, full_name):
    users = oneview_client.security.users.get_all()
    for user in users:
        if user['userName'] == username or user['fullName'] == full_name:
            return True
    return False

# Get the roles associated with a user
user_roles = oneview_client.security.users.get_roles('newUser')

# Get users by their role
def get_users_by_role(role_name):
    all_users = oneview_client.security.users.get_all()
    users_with_role = [user for user in all_users if role_name in user['roles']]
    return users_with_role

# Delete a single user
oneview_client.security.users.delete(retrieved_user)

# Delete multiple users
for user in users_options:
    user_to_delete = oneview_client.security.users.get_by('userName', user['userName'])[0]
    oneview_client.security.users.delete(user_to_delete)
```