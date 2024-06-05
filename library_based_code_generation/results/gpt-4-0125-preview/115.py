```python
from bitcoinlib.encoding import *

# Base conversion examples
examples = [
    ('1a', 16, 10),  # Hexadecimal to Decimal
    ('1010', 2, 16),  # Binary to Hexadecimal
    ('78', 10, 2),  # Decimal to Binary
]

for example in examples:
    original_value, original_base, target_base = example
    converted_value = change_base(original_value, original_base, target_base)
    print(f"{original_value} (base {original_base}) -> {converted_value} (base {target_base})")

# Bitcoin address to public key hash conversion
btc_addresses = [
    '1BoatSLRHtKNngkdXEeobR76b53LETtpyT',
    '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy'
]

for address in btc_addresses:
    pubkey_hash = addr_to_pubkeyhash(address)
    print(f"{address} -> {pubkey_hash.hex()}")

# Public key hash to Bitcoin address conversion
pubkey_hashes = [
    'ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb',
    'f54a5851a9f0e2e5a7ad90b34c9b455080969806b7f96b8c8a3e0f4de8f945ee'
]

for pubkey_hash in pubkey_hashes:
    address = pubkeyhash_to_addr(bytes.fromhex(pubkey_hash))
    print(f"{pubkey_hash} -> {address}")

# Create public key hash from redeem script
redeem_script = '76a91488ac'
redeem_script_bytes = to_bytes(redeem_script, 'hex')
pubkey_hash = hash160(redeem_script_bytes)
print(f"Redeem script hash: {pubkey_hash.hex()}")

# Convert DER encoded signature to a different format
der_sig = '3045022100a34f7f6c8ee5a074e2b3ff1d95d8fc4130e5a4b6e3c3f4e2c43020f890d4426002206d11fbaa17c1814f5ccf9b4af50a48c3db6b4a2d213b212d677785b8a1b24855'
converted_sig = convert_der_sig(der_sig, 'hex')
print(f"Converted signature: {converted_sig}")

# Convert integer to varbyte integer and back
integer = 1000
varbyte_int = int_to_varbyteint(integer)
converted_back_integer = varbyteint_to_int(varbyte_int)
print(f"Integer: {integer} -> Varbyte integer: {varbyte_int.hex()} -> Converted back: {converted_back_integer}")

# Normalize data
data_list = ['Data1', 'data2', 'DATA3']
normalized_strings = [normalize_string(data) for data in data_list]
normalized_vars = [normalize_var(data) for data in data_list]
print(f"Normalized strings: {normalized_strings}")
print(f"Normalized vars: {normalized_vars}")

# Bech32 address to public key hash and back
bech32_address = 'bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq'
pubkey_hash = addr_bech32_to_pubkeyhash(bech32_address)
bech32_converted_back = pubkeyhash_to_addr_bech32(pubkey_hash, prefix='bc')
print(f"Bech32 address: {bech32_address} -> Public key hash: {pubkey_hash.hex()}")
print(f"Public key hash: {pubkey_hash.hex()} -> Bech32 address: {bech32_converted_back}")
```