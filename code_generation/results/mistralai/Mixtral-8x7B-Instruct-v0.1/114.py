 ```python
import os
import json
import logging
from nucypher import Configuration, License, LicenseAgreement, NetworkManager, Policy, Signer
from web3 import Web3
from web3.middleware import geth_poa_middleware
from nucypher.characters.lawful import Alice, Bob
from nucypher.characters.nucypheran import Ursula
from nucypher.crypto.powers import TransactingPower
from nucypher.policy.trevor import TrevorPolicy
from time import sleep

logging.basicConfig(level=logging.INFO)

# Set up environment variables
CONFIG = Configuration()
CONFIG.EthereumNodeEndpoint = os.getenv("ETHEREUM_NODE_ENDPOINT")
CONFIG.WalletFilepath = os.getenv("WALLET_FILEPATH")
CONFIG.AliceAddress = os.getenv("ALICE_ETH_ADDRESS")
CONFIG.AlicePassword = os.getenv("ALICE_WALLET_PASSWORD")

# Connect to Ethereum provider
w3 = Web3(Web3.HTTPProvider(CONFIG.EthereumNodeEndpoint))
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

# Unlock Alice's wallet
with open(CONFIG.WalletFilepath, 'r') as f:
    wallet = w3.eth.account.from_wallet_file(f, CONFIG.AlicePassword)
    alice_account = wallet[CONFIG.AliceAddress]

# Set up Alice's payment method
alice_payment = SubscriptionManagerPayment(alice_account)

# Create an instance of Alice
alice = Alice(
    alice_account.address,
    TransactingPower(alice_account),
    CONFIG.UrsulaThreshold,
    CONFIG.EthereumNodeEndpoint,
    payment_method=alice_payment
)

# Start Alice's learning loop
alice.learn()

# Create a policy label and get the policy public key
policy_label = "heart_rate_data"
policy_public_key = alice.get_policy_public_key(policy_label)

# Generate heart rate samples and save them as a file
heart_rate_samples = generate_heart_rate_samples()
with open("heart_rate_data.json", "w") as f:
    json.dump(heart_rate_samples, f)

# Get the public keys of the recipient (Doctor)
doctor_public_keys = get_doctor_public_keys()

# Create an instance of Bob
bob = Bob(doctor_public_keys)

# Set up policy details
policy_expiration_date = int(time.time()) + 86400  # 1 day from now
m_out_of_n = (1, 1)

# Grant access to Bob by creating a policy and sending it to the NuCypher network
policy = TrevorPolicy(
    policy_public_key,
    policy_label,
    policy_expiration_date,
    m_out_of_n,
    alice. UrsulaCoordinator,
    alice.UrsulaEnclave,
    alice.Signer,
)
policy_data = policy.serialize()
alice.grant(policy_data, bob.public_keys)

# Store additional information about the policy in a JSON file
policy_info = {
    "policy_label": policy_label,
    "policy_public_key": policy_public_key,
    "policy_expiration_date": policy_expiration_date,
    "m_out_of_n": m_out_of_n,
    "doctor_public_keys": doctor_public_keys,
}
with open("policy_info.json", "w") as f:
    json.dump(policy_info, f)
```