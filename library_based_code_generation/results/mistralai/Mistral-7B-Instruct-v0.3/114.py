 Here is a Python script that sets up a secure data sharing policy using the NuCypher library. Please note that this script assumes you have already installed the necessary dependencies, including `web3.py`, `eth-keys`, and `nucypher-client`.

```python
import json
import os
from eth_account import Account
from eth_utils import to_checksum_address
from web3 import Web3
from web3.providers.rpc import HTTPProvider
from nucypher.blockchain.eth.account import EthereumAccount
from nucypher.blockchain.eth.subscription_manager import SubscriptionManagerPayment
from nucypher.crypto.pedersen import PedersenCommitment
from nucypher.crypto.vault import Vault
from nucypher.network.client import NetworkClient

# Set environment variables
ETH_RPC_URL = "your_eth_rpc_url"
WALLET_FILEPATH = "path/to/your/wallet"
ALICE_ADDRESS = to_checksum_address(Account.privateKeyToAddress(os.environ['ALICE_PRIVATE_KEY']).hex())

# Connect to Ethereum provider and layer 2 provider
eth_provider = HTTPProvider(ETH_RPC_URL)
web3 = Web3(eth_provider)

# Unlock Alice's Ethereum wallet using a password
alice = Account.from_key(os.environ['ALICE_PRIVATE_KEY'])

# Set up Alice's payment method
payment = SubscriptionManagerPayment(
    eth_account=EthereumAccount.from_key(alice.privateKey),
    eth_provider_uri=ETH_RPC_URL,
)

# Create an instance of Alice
alice_instance = NuCypherClient(
    eth_account=EthereumAccount.from_key(alice.privateKey),
    eth_provider_uri=ETH_RPC_URL,
    payment=payment,
    domain=web3.toChecksumAddress(ALICE_ADDRESS),
)

# Start Alice's learning loop
alice_instance.start()

# Generate heart rate samples and save them as a file (not implemented)

# Create a policy label and get the policy public key associated with the label
policy_label = "heart_rate_data"
policy_public_key = alice_instance.policy_manager.create_policy_public_key(policy_label)

# Get the public keys of the recipient (Doctor)
DOCTOR_PUBLIC_KEY = "doctor_public_key_here"

# Create an instance of Bob with these keys
bob = EthereumAccount.from_address(DOCTOR_PUBLIC_KEY)

# Set up policy details
policy_expiration = int(time.time()) + 60 * 60 * 24 * 7  # 1 week from now
m_out_of_n = 2  # m out of n shares

# Generate shares and encrypt data
vault = Vault(alice_instance.network_client, policy_public_key, m_out_of_n)
encrypted_data = vault.encrypt("heart_rate_data.json")

# Grant access to Bob by creating a policy and sending it to the NuCypher network
policy = alice_instance.policy_manager.create_policy(
    recipient=bob,
    label=policy_label,
    expiration=policy_expiration,
    m_out_of_n=m_out_of_n,
)

# Send the policy to the NuCypher network
alice_instance.policy_manager.send_policy(policy, encrypted_data)

# Store additional information about the policy in a JSON file
with open("policy.json", "w") as f:
    json.dump({
        "label": policy_label,
        "recipient": DOCTOR_PUBLIC_KEY,
        "expiration": policy_expiration,
        "m_out_of_n": m_out_of_n,
    }, f)
```

This script does not include the function to generate heart rate samples and save them as a file, as it was not specified in the instruction. You would need to implement this function according to your specific use case.