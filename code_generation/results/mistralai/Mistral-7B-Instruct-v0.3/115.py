 ```python
from bitcoinlib.encoding import *

examples = [
    ([17, 101, 49], 10, 16),
    ([255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255], 16, 2),
    ([123], 10, 8),
]

for original, base_orig, base_dest in examples:
    result = change_base(original, base_orig, base_dest)
    print(f"Original: {original}, Base: {base_orig}, Dest Base: {base_dest}, Result: {result}")

bitcoin_addresses = [
    "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
    "3Qc1R5Qw7vX3g4LB6BFB8W2av56zYe33Gf",
    "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"
]

for address in bitcoin_addresses:
    pubkey_hash = addr_to_pubkeyhash(address)
    print(f"Bitcoin Address: {address}, Public Key Hash: {pubkey_hash}")

pubkey_hashes = [
    "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f",
    "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
]

for pubkey_hash in pubkey_hashes:
    address = pubkeyhash_to_addr(pubkey_hash)
    print(f"Public Key Hash: {pubkey_hash}, Bitcoin Address: {address}")

redeem_script = [48, 13, 146, 149, 148, 4, 20, 223, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
public_key_hash = hash160(hash160(redeem_script)[::-1]).to_bytes('hex', byteorder='big')
print(f"Redeem Script: {redeem_script}, Public Key Hash: {public_key_hash}")

der_signature = [4, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 12