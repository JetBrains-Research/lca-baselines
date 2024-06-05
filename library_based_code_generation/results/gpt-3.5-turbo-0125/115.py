```python
from bitcoinlib.encoding import *

base_conversion_examples = [
    ((10, 'A'), 16, 2),
    ((101010, 2), 2, 10),
    ((0xFF, 16), 16, 10)
]

for example in base_conversion_examples:
    original_value, original_base, new_base = example
    result = change_base(original_value, original_base, new_base)
    print(f"Converting {original_value} from base {original_base} to base {new_base}: {result}")

bitcoin_addresses = [
    "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",
    "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"
]

for address in bitcoin_addresses:
    pubkey_hash = addr_to_pubkeyhash(address)
    print(f"Bitcoin Address: {address}, Public Key Hash: {pubkey_hash}")

public_key_hashes = [
    "62e907b15cbf27d5425399ebf6f0fb50ebb88f18",
    "3c176e659bea0f29a3e9bf7880c112b1b31b4dc8"
]

for pubkey_hash in public_key_hashes:
    address = pubkeyhash_to_addr(pubkey_hash)
    print(f"Public Key Hash: {pubkey_hash}, Bitcoin Address: {address}")

redeem_script = b'522102f2b8e0a8b6e6f3f3b1c0e3e6b4e8e5d8f7e3f2b8e0a8b6e6f3f3b1c0e3e6b4e8e5d8f7e3f2b8e0a8b6e6f3f3b1c0e3e6b4e8e5d8f7e3f2103f2b8e0a8b6e6f3f3b1c0e3e6b4e8e5d8f7e3f2b8e0a8b6e6f3f3b1c0e3e6b4e8e5d8f7e3f52ae'

pubkey_hash = hash160(redeem_script)
pubkey_hash_bytes = to_bytes(pubkey_hash)
print(f"Public Key Hash from Redeem Script: {pubkey_hash_bytes.hex()}")

der_signature = b'304402205f8b3b3b7c7b3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b7c3b