 ```python
import json
import os
import random
from datetime import datetime, timedelta

import numpy as np
import web3
from nucypher.characters.lawful import Alice, Bob, Ursula
from nucypher.configs import EthereumNetworks
from nucypher.crypto.powers import TransactingPower
from nucypher.networks.ethereum import EthereumClient
from nucypher.policy import Policy, PolicyParameters
from nucypher.utilities.logging import GlobalLoggingInterceptor, disable_logging, enable_logging
from web3.middleware import geth_poa_middleware

# Set up logging and environment variables
disable_logging()
GlobalLoggingInterceptor.add_interceptor(enable_logging)
os.environ["NUCYPHER_ETH_NETWORK"] = "mainnet"
os.environ["WEB3_PROVIDER_URI"] = "your_ethereum_rpc_endpoint_url"
os.environ["WALLET_FILEPATH"] = "path/to/alice/wallet"
os.environ["ALICE_ETHEREUM_ADDRESS"] = "alice's_ethereum_address"

# Connect to the Ethereum provider and layer 2 provider
ethereum_client = EthereumClient(network=EthereumNetworks.Mainnet)
layer2_provider = ethereum_client.get_layer2_provider()

# Unlock Alice's Ethereum wallet using a password
alice_wallet_filepath = os.environ["WALLET_FILEPATH"]
alice_password = "your_alice_wallet_password"
alice_ethereum_address = os.environ["ALICE_ETHEREUM_ADDRESS"]
alice_web3 = web3.Web3(web3.HTTPProvider(os.environ["WEB3_PROVIDER_URI"]))
alice_web3.middleware_onion.inject(geth_poa_middleware, layer=2)
alice_account = alice_web3.eth.account.from_keyfile(alice_password, alice_wallet_filepath)
alice_address = alice_account.address
assert alice_address == alice_ethereum_address

# Set up Alice's payment method using the SubscriptionManagerPayment class
from nucypher.characters.lawful import SubscriptionManagerPayment

alice_payment = SubscriptionManagerPayment(
    alice_web3,
    alice_account,
    minimum_stake_duration=timedelta(days=30),
    minimum_stake_amount=web3.Web3.toWei(0.1, "ether"),
)

# Create an instance of Alice with her Ethereum address, signer, domain, Ethereum provider URI, and payment method
from nucypher.characters.nucypheran import Nucypheran

alice_signer = alice_account.key
alice_domain = ethereum_client.get_domain()
alice = Nucypheran(
    alice_ethereum_address,
    signer=alice_signer,
    domain=alice_domain,
    ethereum_provider_uri=os.environ["WEB3_PROVIDER_URI"],
    payment=alice_payment,
)

# Start Alice's learning loop
alice.learn()

# Create a policy label and get the policy public key associated with the label
policy_label = "heart_rate_data"
policy_public_key = alice.get_policy_public_key(policy_label)

# Generate heart rate samples using a heart monitor and save them as a file
heart_rate_samples = np.random.randint(60, 120, size=100)
np.save("heart_rate_samples.npy", heart_rate_samples)

# Get the public keys of the recipient (Doctor) and create an instance of Bob with these keys
doctor_public_keys = [
    "doctor_public_key_1",
    "doctor_public_key_2",
]
bob = Bob(doctor_public_keys)

# Set up policy details such as the policy expiration date and m-out-of-n shares
policy_expiration_date = datetime.now() + timedelta(days=30)
m_out_of_n_shares = (1, 1)

# Grant access to Bob by creating a policy and sending it to the NuCypher network
policy_parameters = generate_policy_parameters(
    alice,
    policy_public_key,
    policy_label,
    policy_expiration_date,
    m_out_of_n_shares,
)
policy = create_policy(alice, policy_parameters)
alice.grant(policy, bob)

# Store additional information about the policy in a JSON file
policy_info = {
    "policy_label": policy_label,
    "policy_public_key": policy_public_key,
    "policy_expiration_date": policy_expiration_date,
    "m_out_of_n_shares": m_out_of_n_shares,
}
with open("policy_info.json", "w") as f:
    json.dump(policy_info, f)
```