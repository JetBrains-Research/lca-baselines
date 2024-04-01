#!/bin/bash

# Remote server details
username="tigina"
server="app4.mu4.eqx.labs.intellij.net"

# Remote directory path
remote_dir="/mnt/data/shared-data/lca/bug_localization_data/*"

# Local directory path
local_dir="/Users/Maria.Tigina/PycharmProjects/lca-baselines/data/upd"

key_path="/Users/Maria.Tigina/.ssh/id_rsa_server"
# Secure copy from remote to local
scp -i "$key_path" -r "$username@$server:$remote_dir" "$local_dir"