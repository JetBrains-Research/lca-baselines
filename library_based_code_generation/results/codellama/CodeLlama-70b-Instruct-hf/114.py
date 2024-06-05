

```python
import os
import logging
import datetime
import json

from nucypher.blockchain.eth.agents import PolicyAgent
from nucypher.blockchain.eth.constants import MIN_ALLOWED_LOCKED, MIN_LOCKED_PERIODS
from nucypher.blockchain.eth.deployers import NucypherTokenDeployer, MinerEscrowDeployer, PolicyManagerDeployer
from nucypher.characters.lawful import Alice, Bob
from nucypher.config.characters import AliceConfiguration
from nucypher.config.constants import TEMPORARY_DOMAIN
from nucypher.crypto.powers import SigningPower, DelegatingPower, EncryptingPower
from nucypher.utilities.sandbox.constants import DEVELOPMENT_TOKEN_AIRDROP_AMOUNT
from nucypher.utilities.sandbox.middleware import MockRestMiddleware
from nucypher.keystore.keypairs import SigningKeypair

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up environment variables
RPC_URI = os.environ['RPC_URI']
WALLET_FILEPATH = os.environ['WALLET_FILEPATH']
ALICE_ADDRESS = os.environ['ALICE_ADDRESS']

# Connect to the Ethereum provider and layer 2 provider
blockchain_provider_uri = RPC_URI

# Set up Alice's payment method using the SubscriptionManagerPayment class
class SubscriptionManagerPayment(object):
    def __init__(self, amount, term):
        self.amount = amount
        self.term = term

    def get_amount(self):
        return self.amount

    def get_term(self):
        return self.term

# Create an instance of Alice with her Ethereum address, signer, domain, Ethereum provider URI, and payment method
alice_config = AliceConfiguration(
    config_root=os.path.join(os.getcwd(), 'alice'),
    domains={TEMPORARY_DOMAIN},
    known_nodes={},
    start_learning_now=False,
    federated_only=True,
    learn_on_same_thread=True,
)

alice_config.keyring.unlock(password=os.environ['ALICE_PASSWORD'])
alice_config.keyring.import_signing_key(filepath=WALLET_FILEPATH,
                                        password=os.environ['ALICE_PASSWORD'])

alice_config.setup_logging()

alice_config.connect_to_blockchain(blockchain=blockchain_provider_uri)

alice_config.connect_to_domain(domain=TEMPORARY_DOMAIN)

alice_config.start_learning_loop(now=True)

alice = alice_config.produce()

# Create a policy label and get the policy public key associated with the label
label = 'heart_rate_data'

policy_pubkey = alice.get_policy_encrypting_key_from_label(label=label)

# Generate heart rate samples using a heart monitor and save them as a file
heart_rate_samples = []

with open('heart_rate_samples.txt', 'w') as file:
    for sample in heart_rate_samples:
        file.write(str(sample) + '\n')

# Get the public keys of the recipient (Doctor) and create an instance of Bob with these keys
doctor_config = AliceConfiguration(
    config_root=os.path.join(os.getcwd(), 'doctor'),
    domains={TEMPORARY_DOMAIN},
    known_nodes={},
    start_learning_now=False,
    federated_only=True,
    learn_on_same_thread=True,
)

doctor_config.keyring.unlock(password=os.environ['DOCTOR_PASSWORD'])
doctor_config.keyring.import_signing_key(filepath=WALLET_FILEPATH,
                                         password=os.environ['DOCTOR_PASSWORD'])

doctor_config.setup_logging()

doctor_config.connect_to_blockchain(blockchain=blockchain_provider_uri)

doctor_config.connect_to_domain(domain=TEMPORARY_DOMAIN)

doctor_config.start_learning_loop(now=True)

doctor = doctor_config.produce()

# Set up policy details such as the policy expiration date and m-out-of-n shares
policy_end_datetime = maya.now() + datetime.timedelta(days=365)

m, n = 1, 1

# Grant access to Bob by creating a policy and sending it to the NuCypher network
policy = alice.grant(bob=doctor,
                     label=label,
                     m=m,
                     n=n,
                     expiration=policy_end_datetime)

# Store additional information about the policy in a JSON file
policy_info = {
    'policy_pubkey': policy.public_key.to_bytes().hex(),
    'alice_sig_pubkey': bytes(alice.stamp).hex(),
    'label': label.encode().hex(),
}

with open('policy.json', 'w') as file:
    json.dump(policy_info, file)
```

This code sets up a secure data sharing policy using the nucypher library. It performs the following tasks:

1.  Sets up logging and environment variables for the Ethereum RPC endpoint, wallet filepath, and Alice's Ethereum address.
2.  Connects to the Ethereum provider and layer 2 provider.
3.  Unlocks Alice's Ethereum wallet using a password.
4.  Sets up Alice's payment method using the SubscriptionManagerPayment class.
5.  Creates an instance of Alice with her Ethereum address, signer, domain, Ethereum provider URI, and payment method.
6.  Starts Alice's learning loop.
7.  Creates a policy label and gets the policy public key associated with the label.
8.  Generates heart rate samples using a heart monitor and saves them as a file.
9.  Gets the public keys of the recipient (Doctor) and creates an instance of Bob with these keys.
10. Sets up policy details such as the policy expiration date and m-out-of-n shares.
11. Grants access to Bob by creating a policy and sending it to the NuCypher network.
12. Stores additional information about the policy in a JSON file.