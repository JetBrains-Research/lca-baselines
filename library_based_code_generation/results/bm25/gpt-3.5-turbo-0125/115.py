```python
from bitcoinlib.encoding import *

base_conversion_examples = [
    ((10, 'A'), 16, 2),
    ((101010, 2), 2, 10),
    ((FF, 16), 16, 10)
]

for example in base_conversion_examples:
    original_value, original_base, target_base = example
    result = change_base(original_value, original_base, target_base)
    print(f"Example: {original_value} (base {original_base}) -> {result} (base {target_base})")

bitcoin_addresses = [
    '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
    '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy',
    'bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq'
]

for address in bitcoin_addresses:
    pubkey_hash = addr_to_pubkeyhash(address)
    print(f"Bitcoin Address: {address} -> Public Key Hash: {pubkey_hash}")

public_key_hashes = [
    '62e907b15cbf27d5425399ebf6f0fb50ebb88f18',
    '3c176e659bea0f29a3e9bf7880c112b1b31b4dc8'
]

for pubkey_hash in public_key_hashes:
    address = pubkeyhash_to_addr(pubkey_hash)
    print(f"Public Key Hash: {pubkey_hash} -> Bitcoin Address: {address}")

redeem_script = '522102f2b8e0f379e6f0f0b8b9f3d1e2e1f5b1f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f