```python
from hpeOneView.oneview_client import OneViewClient

config = {
    "ip": "192.168.1.10",
    "credentials": {
        "userName": "admin",
        "password": "password"
    }
}

oneview_client = OneViewClient(config)

# Create a scope
scope = oneview_client.scopes.create({'name': 'Scope1'})

# Create a user with specific permissions
user1 = oneview_client.users.create({'userName': 'user1', 'password': 'password1', 'scopeUris': [scope['uri']]})

# Create multiple users with different permissions
user2 = oneview_client.users.create({'userName': 'user2', 'password': 'password2'})
user3 = oneview_client.users.create({'userName': 'user3', 'password': 'password3'})

# Update user's password
oneview_client.users.update_password(user1['uri'], 'new_password')

# Add a new role to an existing user
oneview_client.users.add_role(user1['uri'], 'NewRole')

# Update the roles of a user
oneview_client.users.update_roles(user1['uri'], ['Role1', 'Role2'])

# Remove certain roles from a user
oneview_client.users.remove_role(user1['uri'], 'Role2')

# Retrieve a user by their username
user = oneview_client.users.get_by_userName('user1')

# Retrieve all users
users = oneview_client.users.get_all()

# Validate if a full name or username is already in use
is_username_in_use = oneview_client.users.validate_username('user1')

# Get the roles associated with a user
user_roles = oneview_client.users.get_roles(user1['uri'])

# Get users by their role
users_by_role = oneview_client.users.get_users_by_role('Role1')

# Delete a single user
oneview_client.users.delete(user1['uri'])

# Delete multiple users
oneview_client.users.delete_multiple([user2['uri'], user3['uri']])
```