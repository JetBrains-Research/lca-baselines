  Here is the Python code based on the instructions:
```
import logging
import os
from nucypher import *

# Set up logging and environment variables
logging.basicConfig(level=logging.INFO)
os.environ["ETHEREUM_RPC_ENDPOINT"] = "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
os.environ["WALLET_FILEPATH"] = "alice_wallet.json"
os.environ["ALICE_ETH_ADDRESS"] = "0x1234567890abcdef"

# Connect to Ethereum provider and layer 2 provider
eth_client = EthereumClient()
layer2_client = Layer2Client()

# Unlock Alice's Ethereum wallet using a password
wallet = Wallet.from_file(os.environ["WALLET_FILEPATH"])
wallet.unlock("password")

# Set up Alice's payment method using the SubscriptionManagerPayment class
payment_method = SubscriptionManagerPayment(eth_client, layer2_client)

# Create an instance of Alice with her Ethereum address, signer, domain, Ethereum provider URI, and payment method
alice = Alice(
    eth_address=os.environ["ALICE_ETH_ADDRESS"],
    signer=wallet.signer,
    domain="example.com",
    eth_provider_uri=os.environ["ETHEREUM_RPC_ENDPOINT"],
    payment_method=payment_method
)

# Start Alice's learning loop
alice.start_learning_loop()

# Create a policy label and get the policy public key associated with the label
policy_label = "heart_rate_policy"
policy_public_key = alice.get_policy_public_key(policy_label)

# Generate heart rate samples using a heart monitor and save them as a file
heart_rate_samples = generate_heart_rate_samples()
with open("heart_rate_samples.csv", "w") as f:
    f.write(heart_rate_samples)

# Get the public keys of the recipient (Doctor) and create an instance of Bob with these keys
doctor_public_key = "0x9876543210fedcba"
bob = Bob(doctor_public_key)

# Set up policy details such as the policy expiration date and m-out-of-n shares
policy_expiration_date = datetime.now() + timedelta(days=30)
m_out_of_n_shares = 2

# Grant access to Bob by creating a policy and sending it to the NuCypher network
policy = create_policy(
    policy_label,
    policy_public_key,
    bob.public_key,
    policy_expiration_date,
    m_out_of_n_shares
)
alice.grant_access(policy)

# Store additional information about the policy in a JSON file
policy_info = {
    "policy_label": policy_label,
    "policy_public_key": policy_public_key,
    "recipient_public_key": bob.public_key,
    "policy_expiration_date": policy_expiration_date,
    "m_out_of_n_shares": m_out_of_n_shares
}
with open("policy_info.json", "w") as f:
    json.dump(policy_info, f)
```