 ```python
from bitcoinlib.encoding import *

examples = [
    ([25], 10, 16),
    ([1024], 16, 8),
    ([16777216], 8, 16),
]

for values, orig_base, new_base in examples:
    result = change_base(values, orig_base, new_base)
    print(f"Original base {orig_base}, new base {new_base}: {values} -> {result}")

bitcoin_addresses = [
    "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    "3Qc1R5Qw7vX3g4LB6BFA7C4qeA7zcR58a8",
    "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"
]

for address in bitcoin_addresses:
    pubkeyhash = addr_to_pubkeyhash(address)
    print(f"Bitcoin address: {address}, Public key hash: {pubkeyhash}")

pubkey_hashes = [
    "00112233445566778899AABBCCDDEEFF",
    "6F6964656E6F7468696E67"
]

for pubkey_hash in pubkey_hashes:
    address = pubkeyhash_to_addr(pubkey_hash)
    print(f"Public key hash: {pubkey_hash}, Bitcoin address: {address}")

redeem_script = b'\x04\x20\x47\x30\x44\x02\x20\x33\x32\x5f\x5b\x65\x72\x6e\x65\x6e\x64\x20\x62\x65\x66\x6f\x72\x20\x63\x6f\x6e\x6e\x65\x63\x74\x69\x6f\x6e'
pkh = hash160(hash160(redeem_script)[:4])
pkh_bytes = pkh.to_bytes(20, byteorder='big')
pkh_hex = pkh_bytes.hex()
print(f"Redeem script: {redeem_script.hex()}, Public key hash (hex): {pkh_hex}")

signature_der = b'\x30\x45\x02\x21\x00\x9f\x9b\x8c\x55\x54\x68\x65\x6c\x6c\x6f\x02\x20\x30\x45\x02\x0e\x03\x42\x00\x47\x30\x44\x02\x20\x30\x2e\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\x04\x20\x30\x0c\x06\x03\x55\x04\x03\x42\x00\