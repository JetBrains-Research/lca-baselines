 Here is a simplified Python code example using the `hpeOneView` library to perform the tasks you've specified. Please note that you'll need to install the `hpeOneView` library and provide your own configuration file.

```python
from hpeOneView.oneview_client import OneViewClient
from hpeOneView.rest_client import RestClient
from hpeOneView.exceptions import OneViewAPIException

# Initialize the client using the configuration file
client = OneViewClient(RestClient(username='<username>', password='<password>',
                                   server='<server>', validate_cert=False))

# Connect to OneView
client.login(username='<admin_username>', password='<admin_password>')

# Create a scope
scope = client.create_scope(name='my_scope')

# Create a user with specific permissions
user_permissions = ['Read-only', 'Manage Users']
user = client.create_user(name='user1', password='password1', scope=scope, permissions=user_permissions)

# Create multiple users with different permissions
user_permissions_2 = ['Read-only', 'Manage Users', 'Manage Networks']
user_2 = client.create_user(name='user2', password='password2', scope=scope, permissions=user_permissions_2)
user_permissions_3 = ['Read-only', 'Manage Servers']
user_3 = client.create_user(name='user3', password='password3', scope=scope, permissions=user_permissions_3)

# Update the user's password
user.update(password='new_password1')

# Add a new role to an existing user
new_role = client.get_role_by_name('Manage Storage')
user.add_role(new_role)

# Update the roles of a user
user.update(permissions=['Read-only', 'Manage Users', 'Manage Storage'])

# Remove certain roles from a user
user.remove_role(client.get_role_by_name('Manage Networks'))

# Retrieve a user by their username
user_by_username = client.get_user_by_name('user1')

# Retrieve all users
all_users = client.get_users()

# Validate if a full name or username is already in use
def is_user_exists(name):
    return any(user.name == name for user in all_users)

# Get the roles associated with a user
roles = user.roles

# Get users by their role
role_name = 'Read-only'
users_by_role = [user for user in all_users if any(role.name == role_name for role in user.roles)]

# Delete a single user
client.delete_user(user_1)

# Delete multiple users
users_to_delete = [user_2, user_3]
for user in users_to_delete:
    client.delete_user(user)
```

This code assumes that you have the necessary permissions to perform these actions in your OneView environment. Please make sure to replace the placeholders with your actual values.