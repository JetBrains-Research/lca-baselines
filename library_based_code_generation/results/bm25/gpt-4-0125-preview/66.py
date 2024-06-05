```python
from hpeOneView.oneview_client import OneViewClient

# Load configuration to connect to OneView
config = {
    "ip": "<OneView_IP>",
    "credentials": {
        "userName": "<username>",
        "password": "<password>"
    }
}

# Establish connection
oneview_client = OneViewClient(config)

# Create a scope
scope_options = {
    "name": "SampleScope",
    "description": "Sample Scope description"
}
scope = oneview_client.scopes.create(scope_options)

# Create a user
user_options = {
    "userName": "sampleUser",
    "password": "password",
    "fullName": "Sample User",
    "emailAddress": "sampleUser@example.com",
    "officePhone": "123-456-7890",
    "mobilePhone": "098-765-4321",
    "roles": ["Read only"],
    "scopeUri": scope['uri']
}
user = oneview_client.security.users.create(user_options)

# Create multiple users with different permissions
users_to_create = [
    {"userName": "user1", "password": "password1", "fullName": "User One", "roles": ["Read only"]},
    {"userName": "user2", "password": "password2", "fullName": "User Two", "roles": ["Infrastructure administrator"]}
]
for user_data in users_to_create:
    oneview_client.security.users.create(user_data)

# Update a user's password
oneview_client.security.users.update_password("sampleUser", "newPassword")

# Add a new role to an existing user
oneview_client.security.users.add_role_to_userName("sampleUser", "Network administrator")

# Update the roles of a user
oneview_client.security.users.update_role_to_userName("sampleUser", ["Read only", "Network administrator"])

# Remove certain roles from a user
oneview_client.security.users.remove_role_from_userName("sampleUser", "Read only")

# Retrieve a user by their username
user_retrieved = oneview_client.security.users.get_by_userName("sampleUser")

# Retrieve all users
all_users = oneview_client.security.users.get_all()

# Validate if a full name or username is already in use
is_in_use = oneview_client.security.users.validate_full_name_or_userName("sampleUser")

# Get the roles associated with a user
user_roles = oneview_client.security.users.get_role_associated_with_userName("sampleUser")

# Get users by their role
users_by_role = oneview_client.security.users.get_user_by_role("Network administrator")

# Delete a single user
oneview_client.security.users.delete("sampleUser")

# Delete multiple users
usernames_to_delete = ["user1", "user2"]
for username in usernames_to_delete:
    oneview_client.security.users.delete(username)
```