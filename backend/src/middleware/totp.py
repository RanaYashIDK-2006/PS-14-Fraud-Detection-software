"""TOTP (Time-based One-Time Password) Authenticator Module.

Implements RFC 6238 TOTP for 30-second code generation.
This provides two-factor authentication for the admin login.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
import time
import base64
import secrets
from typing import Optional


class TOTPAuthenticator:
    """Time-based One-Time Password authenticator."""
    
    def __init__(self, secret: bytes, digits: int = 6, period: int = 30):
        """
        Initialize TOTP authenticator.
        
        Args:
            secret: The shared secret key (bytes)
            digits: Number of digits in the code (default: 6)
            period: Time period in seconds (default: 30)
        """
        self.secret = secret
        self.digits = digits
        self.period = period
    
    def generate_code(self, timestamp: Optional[float] = None) -> str:
        """
        Generate a TOTP code for the given timestamp.
        
        Args:
            timestamp: Unix timestamp (defaults to current time)
            
        Returns:
            The TOTP code as a string
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate time step
        time_step = int(timestamp) // self.period
        
        # Convert time step to bytes (big-endian 8-byte)
        time_bytes = struct.pack('>Q', time_step)
        
        # Calculate HMAC-SHA1
        hmac_hash = hmac.new(self.secret, time_bytes, hashlib.sha1).digest()
        
        # Dynamic truncation
        offset = hmac_hash[-1] & 0x0F
        code_int = struct.unpack('>I', hmac_hash[offset:offset + 4])[0]
        code_int &= 0x7FFFFFFF
        
        # Generate code with specified digits
        code = code_int % (10 ** self.digits)
        
        return str(code).zfill(self.digits)
    
    def verify_code(self, code: str, window: int = 1) -> bool:
        """
        Verify a TOTP code.
        
        Args:
            code: The code to verify
            window: Number of time steps to check before/after current (default: 1)
            
        Returns:
            True if the code is valid, False otherwise
        """
        if not code or len(code) != self.digits:
            return False
        
        current_time = time.time()
        
        # Check current time step and adjacent windows
        for offset in range(-window, window + 1):
            time_step = int(current_time) // self.period + offset
            time_bytes = struct.pack('>Q', time_step)
            
            hmac_hash = hmac.new(self.secret, time_bytes, hashlib.sha1).digest()
            dynamic_offset = hmac_hash[-1] & 0x0F
            code_int = struct.unpack('>I', hmac_hash[dynamic_offset:dynamic_offset + 4])[0]
            code_int &= 0x7FFFFFFF
            
            expected_code = str(code_int % (10 ** self.digits)).zfill(self.digits)
            
            if hmac.compare_digest(code, expected_code):
                return True
        
        return False
    
    def get_time_remaining(self) -> int:
        """
        Get the number of seconds remaining until the current code expires.
        
        Returns:
            Seconds remaining (0-29)
        """
        return self.period - (int(time.time()) % self.period)
    
    def get_secret_base32(self) -> str:
        """
        Get the secret encoded in Base32 (for QR code generation).
        
        Returns:
            Base32-encoded secret
        """
        return base64.b32encode(self.secret).decode('ascii')
    
    @staticmethod
    def generate_secret(length: int = 20) -> bytes:
        """
        Generate a cryptographically secure random secret.
        
        Args:
            length: Length of the secret in bytes (default: 20)
            
        Returns:
            Random secret bytes
        """
        return secrets.token_bytes(length)
    
    @staticmethod
    def from_base32(base32_secret: str) -> 'TOTPAuthenticator':
        """
        Create a TOTP authenticator from a Base32-encoded secret.
        
        Args:
            base32_secret: Base32-encoded secret
            
        Returns:
            TOTPAuthenticator instance
        """
        secret = base64.b32decode(base32_secret)
        return TOTPAuthenticator(secret)


# Convenience function for quick verification
def verify_totp(secret: bytes, code: str, window: int = 1) -> bool:
    """Quick TOTP verification."""
    auth = TOTPAuthenticator(secret)
    return auth.verify_code(code, window)


# Convenience function for quick code generation
def generate_totp(secret: bytes) -> str:
    """Quick TOTP code generation."""
    auth = TOTPAuthenticator(secret)
    return auth.generate_code()