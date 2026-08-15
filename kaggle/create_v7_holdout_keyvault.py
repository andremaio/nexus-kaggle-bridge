#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path('/kaggle/working')
PRIVATE = OUT / 'v7_private_key.pem'
PUBLIC = OUT / 'v7_public_key.pem'
REPORT = OUT / 'v7_keyvault_report.json'


def main() -> None:
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q',
        'cryptography==45.0.6',
    ])
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_der = key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    fingerprint = hashlib.sha256(public_der).hexdigest()
    PRIVATE.write_bytes(private_pem)
    PUBLIC.write_bytes(public_pem)
    os.chmod(PRIVATE, 0o600)
    os.chmod(PUBLIC, 0o644)
    report = {
        'schema': 'nexus.private-holdout-keyvault.v1',
        'algorithm': 'RSA-3072-OAEP-SHA256 + AES-256-GCM',
        'public_key_fingerprint_sha256': fingerprint,
        'private_key_written': True,
        'private_key_printed': False,
        'private_key_uploaded_to_github': False,
        'kernel_must_remain_private': True,
        'automatic_promotion_authorized': False,
    }
    REPORT.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print('NEXUS_V7_KEYVAULT_READY fingerprint=' + fingerprint, flush=True)
    print('NEXUS_V7_PUBLIC_KEY_BEGIN', flush=True)
    print(public_pem.decode('ascii').strip(), flush=True)
    print('NEXUS_V7_PUBLIC_KEY_END', flush=True)
    print('NEXUS_V7_PRIVATE_KEY_PRINTED=false', flush=True)


if __name__ == '__main__':
    main()
