import os
import logging
from nucypher.characters.lawful import Alice, Bob
from nucypher.config import Config
from nucypher.network.middleware import RestMiddleware
from nucypher.crypto.kits import UmbralMessageKit
from nucypher.data_sources import DataSource
from nucypher.policy import Policy
from nucypher.data_sources import DataSource
from nucypher.network.middleware import RestMiddleware
from nucypher.policy import EncryptingPolicy
from nucypher.characters import Character
from nucypher.network.middleware import RestMiddleware
from nucypher.policy import EncryptingPolicy
from nucypher.characters.lawful import Ursula
from nucypher.crypto.powers import SigningPower, DecryptingPower
from nucypher.crypto.kits import UmbralMessageKit
from nucypher.data_sources import DataSource
from nucypher.policy import Policy
from nucypher.data_sources import DataSource
from nucypher.network.middleware import RestMiddleware
from nucypher.policy import EncryptingPolicy
from nucypher.characters import Character
from nucypher.network.middleware import RestMiddleware
from nucypher.policy import EncryptingPolicy
from nucypher.characters.lawful import Ursula
from nucypher.crypto.powers import SigningPower, DecryptingPower
from nucypher.crypto.kits import UmbralMessageKit
from nucypher.data_sources import DataSource
from nucypher.policy import Policy

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set environment variables
os.environ["ETH_RPC_ENDPOINT"] = "http://localhost:8545"
os.environ["WALLET_FILEPATH"] = "/path/to/wallet.json"
os.environ["ALICE_ADDRESS"] = "0x1234567890abcdef"

# Connect to Ethereum provider and layer 2 provider
config = Config()
config.initialize(network_middleware=RestMiddleware(),
                 ethereum_rpc_endpoint=os.environ["ETH_RPC_ENDPOINT"])

# Unlock Alice's Ethereum wallet
alice = Alice(config)
alice.unlock(password="password")

# Set up Alice's payment method
from nucypher.characters.lawful import Ursula
from nucypher.crypto.powers import SigningPower, DecryptingPower
from nucypher.crypto.kits import UmbralMessageKit
from nucypher.data_sources import DataSource
from nucypher.policy import Policy

# Create an instance of Alice
alice = Alice(
    config,
    federated_only=True,
    known_nodes={ursula},
    start_learning_now=True,
    federated_only=True,
    learn_on_same_thread=True,
    abort_on_learning_error=True,
    domain="example.com",
    provider_uri="http://localhost:11501",
    network_middleware=RestMiddleware(),
    save_metadata=False,
    deposit=0
)

# Start Alice's learning loop
alice.start_learning_loop(now=True)

# Create policy label and get policy public key
label = b'Heart Rate Data'
policy_pubkey = alice.get_policy_pubkey_from_label(label)

# Generate heart rate samples and save to file
heart_rate_samples = [60, 65, 70, 75, 80]
with open('heart_rate_samples.txt', 'w') as file:
    for sample in heart_rate_samples:
        file.write(str(sample) + '\n')

# Get public keys of recipient (Doctor) and create instance of Bob
doctor_pubkey = b'0xabcdef1234567890'
bob = Bob(
    config,
    known_nodes={ursula},
    start_learning_now=True,
    federated_only=True,
    learn_on_same_thread=True,
    abort_on_learning_error=True,
    domain="example.com",
    provider_uri="http://localhost:11501",
    network_middleware=RestMiddleware(),
    save_metadata=False,
    deposit=0
)

# Set up policy details
policy_expiration = 1000
m = 2
n = 3

# Grant access to Bob by creating policy and sending to NuCypher network
policy = alice.grant(bob, label, m=m, n=n, expiration=policy_expiration)

# Store policy information in JSON file
policy_info = {
    "label": label.decode(),
    "policy_pubkey": policy_pubkey.to_bytes().hex(),
    "doctor_pubkey": doctor_pubkey.decode(),
    "policy_expiration": policy_expiration,
    "m": m,
    "n": n
}
with open('policy_info.json', 'w') as file:
    json.dump(policy_info, file)