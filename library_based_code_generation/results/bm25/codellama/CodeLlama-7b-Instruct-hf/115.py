  Here is the Python code based on the given instructions:
```
from bitcoinlib.encoding import *

# Define a list of examples for base conversion
examples = [
    (12345, 10, 2),
    (12345, 2, 10),
    (12345, 16, 10),
    (12345, 10, 16),
    (12345, 2, 16),
    (12345, 16, 2),
    (12345, 10, 8),
    (12345, 8, 10),
    (12345, 16, 8),
    (12345, 8, 16)
]

# Iterate over the list and print each example and its result after conversion
for example in examples:
    original_value, original_base, target_base = example
    converted_value = change_base(original_value, original_base, target_base)
    print(f"{original_value} in base {original_base} is {converted_value} in base {target_base}")

# Demonstrate the conversion of Bitcoin addresses to public key hashes
bitcoin_addresses = [
    "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    "1BpbpfLdY7oBS9gK7aDXgvMgr1DpvNhEB2",
    "1C1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
]

for address in bitcoin_addresses:
    public_key_hash = addr_to_pubkeyhash(address)
    print(f"{address} corresponds to public key hash {public_key_hash}")

# Demonstrate the conversion from public key hashes to Bitcoin addresses
public_key_hashes = [
    "0000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000001",
    "0000000000000000000000000000000000000002"
]

for public_key_hash in public_key_hashes:
    address = pubkeyhash_to_addr(public_key_hash)
    print(f"{public_key_hash} corresponds to Bitcoin address {address}")

# Demonstrate the creation of a public key hash from a redeem script
redeem_script = "76a9141234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01234567890abcdef01