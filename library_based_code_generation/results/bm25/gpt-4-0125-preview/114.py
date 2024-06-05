```python
import os
import json
import logging
from datetime import datetime, timedelta
from nucypher.blockchain.eth.interfaces import BlockchainInterfaceFactory
from nucypher.blockchain.eth.signers import Signer
from nucypher.characters.lawful import Alice, Bob
from nucypher.config.constants import TEMPORARY_DOMAIN
from nucypher.crypto.powers import SigningPower, EncryptingPower
from nucypher.network.middleware import RestMiddleware
from nucypher.utilities.logging import SimpleObserver
from umbral.keys import UmbralPublicKey

# 1. Set up logging and environment variables
logging.basicConfig(level=logging.DEBUG)
observer = SimpleObserver()
logging.getLogger('nucypher').addObserver(observer)

ETH_PROVIDER_URI = os.getenv('ETH_PROVIDER_URI')
WALLET_FILEPATH = os.getenv('WALLET_FILEPATH')
ALICE_ETH_ADDRESS = os.getenv('ALICE_ETH_ADDRESS')

# 2. Connect to the Ethereum provider and layer 2 provider
BlockchainInterfaceFactory.initialize_interface(provider_uri=ETH_PROVIDER_URI)

# 3. Unlock Alice's Ethereum wallet
password = input("Enter Alice's wallet password: ")
signer = Signer.from_signer_uri(WALLET_FILEPATH)
signer.unlock_account(ALICE_ETH_ADDRESS, password)

# 4. Set up Alice's payment method
# Assuming SubscriptionManagerPayment class is available and set up is outside the scope of this code snippet

# 5. Create an instance of Alice
domain = TEMPORARY_DOMAIN
alice = Alice(
    known_nodes=[],
    domain=domain,
    ethereum_address=ALICE_ETH_ADDRESS,
    signer=signer,
    provider_uri=ETH_PROVIDER_URI,
    # Assuming payment_method is an instance of SubscriptionManagerPayment
    # payment_method=payment_method
)

# 6. Start Alice's learning loop
alice.start_learning_loop(now=True)

# 7. Create a policy label and get the policy public key
policy_label = "heart-rate-data"
policy_pubkey = alice.get_policy_encrypting_key_from_label(policy_label)

# 8. Generate heart rate samples and save them as a file
# This is a placeholder for the actual heart rate monitor data generation
heart_rate_data = "80,82,78,77,75,76,74,73"
with open("heart_rate_data.txt", "w") as f:
    f.write(heart_rate_data)

# 9. Get the public keys of the recipient (Doctor) and create an instance of Bob
doctor_pubkeys = {'enc': UmbralPublicKey.from_bytes(b"<doctor's encrypting pubkey>"),
                  'sig': UmbralPublicKey.from_bytes(b"<doctor's signing pubkey>")}
doctor = Bob(
    known_nodes=[],
    domain=domain,
    encrypting_key=doctor_pubkeys['enc'],
    signing_key=doctor_pubkeys['sig'],
    provider_uri=ETH_PROVIDER_URI
)

# 10. Set up policy details
policy_end_datetime = datetime.now() + timedelta(days=5)
m, n = 2, 3  # m-out-of-n shares

# 11. Grant access to Bob by creating a policy
policy = alice.grant(
    bob=doctor,
    label=policy_label,
    m=m,
    n=n,
    expiration=policy_end_datetime
)

# 12. Store additional information about the policy in a JSON file
policy_info = {
    'policy_label': policy_label,
    'policy_pubkey': policy_pubkey.to_bytes().hex(),
    'alice_sig_pubkey': bytes(alice.stamp).hex(),
    'policy_expiration': policy_end_datetime.isoformat(),
    'm': m,
    'n': n
}
with open("policy_info.json", "w") as f:
    json.dump(policy_info, f)
```