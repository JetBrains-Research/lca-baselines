```python
import os
import json
import logging
from nucypher.characters.lawful import Alice, Bob
from nucypher.config.characters import AliceConfiguration
from nucypher.network.middleware import RestMiddleware
from nucypher.crypto.powers import SigningPower, DecryptingPower
from nucypher.blockchain.eth.signers import Signer
from nucypher.blockchain.eth.interfaces import BlockchainInterfaceFactory
from nucypher.blockchain.eth.token import NU
from nucypher.blockchain.eth.agents import NucypherTokenAgent, PolicyAgent
from datetime import datetime, timedelta
from umbral.keys import UmbralPublicKey
import maya

# Setup logging
logging.basicConfig(level=logging.INFO)

# Environment variables
os.environ['NUCYPHER_ETH_PROVIDER_URI'] = 'YOUR_ETH_PROVIDER_URI'
os.environ['NUCYPHER_WALLET_FILEPATH'] = 'path/to/your/wallet.json'
os.environ['NUCYPHER_ALICE_ETH_ADDRESS'] = 'your_alice_eth_address'
wallet_password = 'your_wallet_password'

# Connect to the Ethereum provider
BlockchainInterfaceFactory.initialize_interface(provider_uri=os.environ['NUCYPHER_ETH_PROVIDER_URI'])

# Unlock Alice's Ethereum wallet
signer = Signer.from_signer_uri(uri=os.environ['NUCYPHER_WALLET_FILEPATH'], password=wallet_password)

# Setup Alice's payment method
# Assuming SubscriptionManagerPayment is a placeholder for actual payment setup, which might involve the NucypherTokenAgent or similar
# This part of the code is pseudo-code as the actual implementation details for setting up payments are not provided
token_agent = NucypherTokenAgent()
policy_agent = PolicyAgent(token_agent=token_agent)
payment_method = "SubscriptionManagerPayment"  # Placeholder for actual payment method setup

# Create an instance of Alice
domain = 'my_domain'
alice = Alice(
    known_nodes=[],
    domain=domain,
    signer=signer,
    provider_uri=os.environ['NUCYPHER_ETH_PROVIDER_URI'],
    federated_only=False,
    checksum_address=os.environ['NUCYPHER_ALICE_ETH_ADDRESS']
)

# Start Alice's learning loop
alice.start_learning_loop(now=True)

# Create a policy label and get the policy public key
policy_label = "heart-rate-data"
policy_pubkey = alice.get_policy_encrypting_key_from_label(policy_label)

# Generate heart rate samples and save them as a file
heart_rate_samples = [72, 75, 71, 73, 78]  # Example heart rate samples
with open('heart_rate_samples.txt', 'w') as f:
    for sample in heart_rate_samples:
        f.write(f"{sample}\n")

# Get the public keys of the recipient (Doctor) and create an instance of Bob
doctor_pubkeys = {'enc': UmbralPublicKey.from_bytes(b'...'), 'sig': UmbralPublicKey.from_bytes(b'...')}
doctor = Bob.from_public_keys(
    verifying_key=doctor_pubkeys['sig'],
    encrypting_key=doctor_pubkeys['enc'],
    federated_only=False
)

# Set up policy details
policy_end_datetime = maya.now() + timedelta(days=5)
m, n = 2, 3  # m-out-of-n

# Grant access to Bob by creating a policy
policy = alice.grant(
    bob=doctor,
    label=policy_label,
    m=m,
    n=n,
    expiration=policy_end_datetime
)

# Store additional information about the policy in a JSON file
policy_info = {
    'policy_label': policy_label,
    'policy_pubkey': policy_pubkey.to_bytes().hex(),
    'doctor_pubkeys': {
        'enc': doctor_pubkeys['enc'].to_bytes().hex(),
        'sig': doctor_pubkeys['sig'].to_bytes().hex()
    },
    'policy_expiration': policy_end_datetime.rfc3339(),
    'm': m,
    'n': n
}

with open('policy_info.json', 'w') as f:
    json.dump(policy_info, f)
```