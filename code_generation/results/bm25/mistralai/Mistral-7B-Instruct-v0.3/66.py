 Here is a Python script that uses the `hpeOneView` library to perform the tasks mentioned in your instruction. Please make sure to install the library using `pip install hpeOneView`. Also, note that you need to have a valid configuration file (`oneview.ini`) in the same directory as the script.

```python
from hpeOneView.OneViewClient import OneViewClient
from hpeOneView.OneViewException import OneViewException

client = OneViewClient.from_environment_variables()

# Create a scope
scope = client.create_scope(name='MyScope')

# Define user permissions
read_only_permissions = ['Read-Only']
read_write_permissions = ['Read-Only', 'Read-Write']
admin_permissions = ['Read-Only', 'Read-Write', 'Admin']

# Create users with specific permissions
users = []
users.append(client.create_user(
    name='User1',
    password='User1Password',
    email='user1@example.com',
    scope=scope,
    permissions=read_only_permissions
))
users.append(client.create_user(
    name='User2',
    password='User2Password',
    email='user2@example.com',
    scope=scope,
    permissions=read_write_permissions
))
users.append(client.create_user(
    name='User3',
    password='User3Password',
    email='user3@example.com',
    scope=scope,
    permissions=admin_permissions
))

# Update user's password
client.update_user(users[0].id, password='UpdatedUser1Password')

# Add a new role to an existing user
client.add_role_to_user(users[0].id, 'NewRole')

# Update the roles of a user
client.update_user(users[1].id, permissions=admin_permissions)

# Remove certain roles from a user
client.remove_role_from_user(users[2].id, 'NewRole')

# Retrieve a user by their username
user = client.get_user_by_username('User1')

# Retrieve all users
all_users = client.get_users()

# Validate if a full name or username is already in use
try:
    client.create_user(name='User1', password='User1Password')
    print("Username or Full Name is already in use")
except OneViewException as e:
    print(e.message)

# Get the roles associated with a user
roles = client.get_role_associated_with_user(users[0].id)

# Get users by their role
users_by_role = client.get_users_by_role('Admin')

# Delete a single user
client.delete_user(users[1].id)

# Delete multiple users
client.delete_multiple_users([users[0].id, users[2].id])
```

This script creates a scope, defines three users with different permissions, updates the password of the first user, adds a new role to the first user, updates the roles of the second user, removes a role from the third user, retrieves a user by their username, retrieves all users, validates if a full name or username is already in use, gets the roles associated with a user, gets users by their role, and deletes a single user and multiple users.