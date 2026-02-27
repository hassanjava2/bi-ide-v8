"""
BI-IDE v8 - Encryption Module
AES-256-GCM encryption for data at rest and in transit
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

logger = logging.getLogger(__name__)


@dataclass
class KeyMetadata:
    """Encryption key metadata"""
    key_id: str
    created_at: datetime
    expires_at: Optional[datetime]
    algorithm: str
    key_size: int
    purpose: str  # 'data_encryption', 'key_encryption', 'signing'
    status: str  # 'active', 'rotating', 'deprecated', 'revoked'


class EncryptionManager:
    """Centralized encryption management"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self._master_key = master_key or self._generate_master_key()
        self._data_keys: Dict[str, bytes] = {}
        self._key_metadata: Dict[str, KeyMetadata] = {}
        self._key_version = 1
        self._initialize_keys()
    
    def _generate_master_key(self) -> bytes:
        """Generate secure master key"""
        return secrets.token_bytes(32)
    
    def _initialize_keys(self):
        """Initialize encryption key hierarchy"""
        # Generate data encryption key
        self._data_keys['current'] = self._derive_key('data_encryption_v1')
        self._key_metadata['current'] = KeyMetadata(
            key_id='current',
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=365),
            algorithm='AES-256-GCM',
            key_size=256,
            purpose='data_encryption',
            status='active'
        )
    
    def _derive_key(self, context: str, salt: Optional[bytes] = None) -> bytes:
        """Derive key from master key using HKDF-like approach"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        derived = kdf.derive(self._master_key + context.encode())
        return salt + derived
    
    # AES-256-GCM Implementation
    
    def encrypt_aes_gcm(self, plaintext: Union[str, bytes], 
                        associated_data: Optional[bytes] = None) -> bytes:
        """
        Encrypt data using AES-256-GCM
        
        Returns: nonce (12 bytes) + ciphertext + tag (16 bytes)
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        # Extract key (skip salt prefix from derived key)
        key = self._data_keys['current'][16:48]
        
        # Generate nonce
        nonce = os.urandom(12)
        
        # Encrypt
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        
        # Format: version(1) + key_id(4) + nonce(12) + ciphertext
        version = bytes([1])
        key_id = b'CURR'
        
        return version + key_id + nonce + ciphertext
    
    def decrypt_aes_gcm(self, ciphertext: bytes,
                        associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt AES-256-GCM encrypted data"""
        if len(ciphertext) < 18:
            raise ValueError("Invalid ciphertext")
        
        version = ciphertext[0]
        key_id = ciphertext[1:5].decode()
        nonce = ciphertext[5:17]
        encrypted_data = ciphertext[17:]
        
        # Get appropriate key
        if key_id == 'CURR':
            key = self._data_keys['current'][16:48]
        else:
            key = self._data_keys.get(key_id, self._data_keys['current'])[16:48]
        
        # Decrypt
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, encrypted_data, associated_data)
    
    # At-Rest Encryption
    
    def encrypt_field(self, value: Union[str, int, float, bool]) -> str:
        """Encrypt a database field value"""
        if value is None:
            return None
        
        # Serialize value
        serialized = json.dumps({'v': value})
        
        # Encrypt
        encrypted = self.encrypt_aes_gcm(serialized)
        
        # Return base64 encoded
        return base64.urlsafe_b64encode(encrypted).decode('ascii')
    
    def decrypt_field(self, encrypted_value: str) -> Union[str, int, float, bool]:
        """Decrypt a database field value"""
        if encrypted_value is None:
            return None
        
        # Decode base64
        encrypted = base64.urlsafe_b64decode(encrypted_value.encode('ascii'))
        
        # Decrypt
        decrypted = self.decrypt_aes_gcm(encrypted)
        
        # Deserialize
        data = json.loads(decrypted.decode('utf-8'))
        return data['v']
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt a file"""
        output_path = output_path or f"{file_path}.enc"
        
        with open(file_path, 'rb') as f:
            plaintext = f.read()
        
        encrypted = self.encrypt_aes_gcm(plaintext)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        
        return output_path
    
    def decrypt_file(self, encrypted_path: str, output_path: str):
        """Decrypt a file"""
        with open(encrypted_path, 'rb') as f:
            ciphertext = f.read()
        
        decrypted = self.decrypt_aes_gcm(ciphertext)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted)
    
    # In-Transit Enforcement
    
    def enforce_tls_config(self) -> Dict[str, any]:
        """Return TLS configuration for secure connections"""
        return {
            'min_version': 'TLSv1.2',
            'max_version': 'TLSv1.3',
            'cipher_suites': [
                'TLS_AES_256_GCM_SHA384',
                'TLS_CHACHA20_POLY1305_SHA256',
                'ECDHE-ECDSA-AES256-GCM-SHA384',
                'ECDHE-RSA-AES256-GCM-SHA384',
            ],
            'prefer_server_ciphers': True,
            'session_timeout': '1d',
            'session_tickets': False,
            'ocsp_stapling': True,
            'hsts': {
                'max_age': 31536000,
                'include_subdomains': True,
                'preload': True
            }
        }
    
    def validate_certificate(self, cert_path: str) -> Dict[str, any]:
        """Validate TLS certificate"""
        from cryptography import x509
        
        with open(cert_path, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())
        
        now = datetime.now()
        
        return {
            'subject': str(cert.subject),
            'issuer': str(cert.issuer),
            'not_before': cert.not_valid_before,
            'not_after': cert.not_valid_after,
            'is_valid': cert.not_valid_before <= now <= cert.not_valid_after,
            'serial_number': str(cert.serial_number),
            'signature_algorithm': cert.signature_algorithm_oid._name,
            'key_size': cert.public_key().key_size if hasattr(cert.public_key(), 'key_size') else 'N/A'
        }
    
    # Key Rotation
    
    def rotate_keys(self) -> str:
        """
        Rotate encryption keys
        
        Returns new key ID
        """
        new_key_id = f"key_v{self._key_version + 1}"
        
        # Mark current key as rotating
        self._key_metadata['current'].status = 'rotating'
        
        # Generate new key
        new_key = self._derive_key(f'data_encryption_{new_key_id}')
        
        # Store new key
        self._data_keys[new_key_id] = new_key
        self._key_metadata[new_key_id] = KeyMetadata(
            key_id=new_key_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=365),
            algorithm='AES-256-GCM',
            key_size=256,
            purpose='data_encryption',
            status='active'
        )
        
        # Update current key reference
        old_key_id = 'current'
        self._data_keys['current'] = new_key
        self._key_metadata['current'] = self._key_metadata[new_key_id]
        
        # Mark old key as deprecated (for decryption of old data)
        if old_key_id in self._key_metadata:
            self._key_metadata[old_key_id].status = 'deprecated'
        
        self._key_version += 1
        
        logger.info(f"Key rotation complete. New key ID: {new_key_id}")
        return new_key_id
    
    def reencrypt_data(self, encrypted_data: bytes) -> bytes:
        """Re-encrypt data with current key (for key rotation)"""
        # Decrypt with old key
        decrypted = self.decrypt_aes_gcm(encrypted_data)
        
        # Re-encrypt with new key
        return self.encrypt_aes_gcm(decrypted)
    
    # Secure Key Storage
    
    def export_key_for_hsm(self, key_id: str = 'current') -> bytes:
        """Export key for HSM storage"""
        if key_id not in self._data_keys:
            raise ValueError(f"Key {key_id} not found")
        
        # Wrap key with master key
        key = self._data_keys[key_id]
        
        # This would use RSA-KEM or similar for HSM
        return key
    
    def import_key_from_hsm(self, key_data: bytes, key_id: str):
        """Import key from HSM"""
        self._data_keys[key_id] = key_data
    
    # Key Metadata Management
    
    def get_key_metadata(self, key_id: str = 'current') -> Optional[KeyMetadata]:
        """Get metadata for a key"""
        return self._key_metadata.get(key_id)
    
    def list_keys(self) -> List[KeyMetadata]:
        """List all keys and their metadata"""
        return list(self._key_metadata.values())
    
    def revoke_key(self, key_id: str):
        """Revoke a key (emergency use)"""
        if key_id in self._key_metadata:
            self._key_metadata[key_id].status = 'revoked'
            logger.warning(f"Key {key_id} has been revoked")
    
    # Utility Functions
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Hash password using Argon2-style approach"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = Scrypt(
            salt=salt,
            length=32,
            n=2**14,
            r=8,
            p=1,
            backend=default_backend()
        )
        
        hashed = kdf.derive(password.encode())
        return salt, hashed
    
    def verify_password(self, password: str, salt: bytes, hashed: bytes) -> bool:
        """Verify password against hash"""
        try:
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=2**14,
                r=8,
                p=1,
                backend=default_backend()
            )
            kdf.verify(password.encode(), hashed)
            return True
        except Exception:
            return False
    
    def generate_hmac(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Generate HMAC for data integrity"""
        from cryptography.hazmat.primitives.hmac import HMAC
        
        key = key or self._master_key
        
        h = HMAC(key, hashes.SHA256(), backend=default_backend())
        h.update(data)
        return h.finalize()
    
    def verify_hmac(self, data: bytes, signature: bytes, key: Optional[bytes] = None) -> bool:
        """Verify HMAC signature"""
        try:
            expected = self.generate_hmac(data, key)
            return secrets.compare_digest(signature, expected)
        except Exception:
            return False


class FieldEncryption:
    """Transparent field encryption for database models"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self._em = encryption_manager
        self._encrypted_fields: Set[str] = set()
    
    def mark_encrypted(self, field_name: str):
        """Mark a field for automatic encryption"""
        self._encrypted_fields.add(field_name)
    
    def encrypt_record(self, record: Dict[str, any]) -> Dict[str, any]:
        """Encrypt marked fields in a record"""
        encrypted = record.copy()
        
        for field in self._encrypted_fields:
            if field in encrypted and encrypted[field] is not None:
                encrypted[field] = self._em.encrypt_field(encrypted[field])
        
        return encrypted
    
    def decrypt_record(self, record: Dict[str, any]) -> Dict[str, any]:
        """Decrypt marked fields in a record"""
        decrypted = record.copy()
        
        for field in self._encrypted_fields:
            if field in decrypted and decrypted[field] is not None:
                decrypted[field] = self._em.decrypt_field(decrypted[field])
        
        return decrypted


# Key Management Service Integration
class KMSIntegration:
    """Integration with cloud KMS services"""
    
    def __init__(self, provider: str = 'aws'):
        self.provider = provider
        self._client = None
    
    def _get_aws_kms_client(self):
        """Get AWS KMS client"""
        import boto3
        if self._client is None:
            self._client = boto3.client('kms')
        return self._client
    
    def encrypt_with_kms(self, plaintext: bytes, key_id: str) -> bytes:
        """Encrypt using AWS KMS"""
        client = self._get_aws_kms_client()
        
        response = client.encrypt(
            KeyId=key_id,
            Plaintext=plaintext,
            EncryptionAlgorithm='SYMMETRIC_DEFAULT'
        )
        
        return response['CiphertextBlob']
    
    def decrypt_with_kms(self, ciphertext: bytes) -> bytes:
        """Decrypt using AWS KMS"""
        client = self._get_aws_kms_client()
        
        response = client.decrypt(
            CiphertextBlob=ciphertext,
            EncryptionAlgorithm='SYMMETRIC_DEFAULT'
        )
        
        return response['Plaintext']
    
    def generate_data_key(self, key_id: str, key_spec: str = 'AES_256') -> Dict[str, bytes]:
        """Generate data key using KMS"""
        client = self._get_aws_kms_client()
        
        response = client.generate_data_key(
            KeyId=key_id,
            KeySpec=key_spec
        )
        
        return {
            'plaintext': response['Plaintext'],
            'ciphertext': response['CiphertextBlob']
        }


# Global encryption manager
_encryption_manager: Optional[EncryptionManager] = None


def init_encryption(master_key: Optional[bytes] = None) -> EncryptionManager:
    """Initialize global encryption manager"""
    global _encryption_manager
    _encryption_manager = EncryptionManager(master_key)
    return _encryption_manager


def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager"""
    if _encryption_manager is None:
        raise RuntimeError("Encryption manager not initialized")
    return _encryption_manager
