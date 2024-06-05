  from hpeOneView import OneViewClient

# Load configuration from file
config = OneViewClient.load_config()

# Create a new scope
scope = OneViewClient.create_scope(config, "MyScope")

# Create a new user with specific permissions
user = OneViewClient.create_user(config, "MyUser", "MyPassword", ["Login", "User.Read"])

# Create multiple users with different permissions
users = [
    OneViewClient.create_user(config, "User1", "Password1", ["Login", "User.Read"]),
    OneViewClient.create_user(config, "User2", "Password2", ["Login", "User.Read", "User.Write"]),
    OneViewClient.create_user(config, "User3", "Password3", ["Login", "User.Read", "User.Write", "User.Delete"])
]

# Update user's password
OneViewClient.update_user_password(config, user["username"], "NewPassword")

# Add a new role to an existing user
OneViewClient.add_user_role(config, user["username"], "MyRole")

# Update the roles of a user
OneViewClient.update_user_roles(config, user["username"], ["Login", "User.Read", "User.Write"])

# Remove certain roles from a user
OneViewClient.remove_user_roles(config, user["username"], ["User.Read"])

# Retrieve a user by their username
user = OneViewClient.get_user_by_username(config, "MyUser")

# Retrieve all users
users = OneViewClient.get_all_users(config)

# Validate if a full name or username is already in use
is_in_use = OneViewClient.is_username_in_use(config, "MyUser")

# Get the roles associated with a user
roles = OneViewClient.get_user_roles(config, user["username"])

# Get users by their role
users = OneViewClient.get_users_by_role(config, "MyRole")

# Delete a single user
OneViewClient.delete_user(config, user["username"])

# Delete multiple users
OneViewClient.delete_users(config, users)