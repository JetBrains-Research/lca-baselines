```python
from bitcoinlib.encoding import *

# Base conversion examples
base_conversion_examples = [
    ('1e', 16, 10),  # Hexadecimal to Decimal
    ('111', 2, 10),  # Binary to Decimal
    ('123', 10, 16),  # Decimal to Hexadecimal
    ('7b', 16, 2),  # Hexadecimal to Binary
]

for value, base_from, base_to in base_conversion_examples:
    result = change_base(value, base_from, base_to)
    print(f"{value} (base {base_from}) -> {result} (base {base_to})")

# Bitcoin addresses to public key hashes
btc_addresses = [
    '1BoatSLRHtKNngkdXEeobR76b53LETtpyT',
    '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy'
]

for address in btc_addresses:
    pubkey_hash = addr_to_pubkeyhash(address)
    print(f"{address} -> {pubkey_hash}")

# Public key hashes to Bitcoin addresses
pubkey_hashes = [
    '89ABCDEFABBAABBAABBAABBAABBAABBAABBAABBA',
    'FEDCBA98765432100123456789ABCDEFABBAABBA'
]

for pubkey_hash in pubkey_hashes:
    address = pubkeyhash_to_addr(pubkey_hash)
    print(f"{pubkey_hash} -> {address}")

# Create public key hash from redeem script
redeem_script = '76a91489abcdefabbaabbaabbaabbaabbaabbaabba88ac'
redeem_script_bytes = to_bytes(redeem_script, 'hex')
pubkey_hash = hash160(redeem_script_bytes)
print(f"Redeem script hash: {pubkey_hash.hex()}")

# Convert DER encoded signature
der_encoded_sig = '3045022100abcdef1234567890abcdef1234567890abcdef1234567890abcdef123456789022034567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef123456'
converted_sig = convert_der_sig(der_encoded_sig, 'hex')
print(f"Converted signature: {converted_sig}")

# Convert integer to varbyte integer and back
integer = 123456
varbyte_int = int_to_varbyteint(integer)
print(f"Integer to varbyte: {varbyte_int.hex()}")
converted_back_int = varbyteint_to_int(varbyte_int)
print(f"Varbyte to integer: {converted_back_int}")

# Normalize data
data_list = ['Data1', 'data2', 'DATA3']
normalized_strings = [normalize_string(data) for data in data_list]
normalized_vars = [normalize_var(data) for data in data_list]
print(f"Normalized strings: {normalized_strings}")
print(f"Normalized vars: {normalized_vars}")

# Bech32 address to public key hash and back
bech32_address = 'bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq'
pubkey_hash = addr_bech32_to_pubkeyhash(bech32_address)
print(f"{bech32_address} -> {pubkey_hash.hex()}")

pubkey_hash = '0014751e76e8199196d454941c45d1b3a323f1433bd6'
bech32_address = pubkeyhash_to_addr_bech32(pubkey_hash, prefix='bc')
print(f"{pubkey_hash} -> {bech32_address}")
```