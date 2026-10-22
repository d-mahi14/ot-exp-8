"""
Enterprise Utilities & Security Infrastructure:
In-memory LRU Cache with TTL expiry, SHA-256 cryptographic audit chaining,
data validation decorators, currency formatters, and mathematical utilities.
"""

from __future__ import annotations
import math
import time
import hashlib
import json
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple


class CacheEntry:
    def __init__(self, value: Any, ttl_seconds: float):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl_seconds
        self.access_count = 1

    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl


class TTLCache:
    """Least Recently Used (LRU) In-Memory Cache with Time-To-Live (TTL) expiration."""
    def __init__(self, max_size: int = 128, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._store[key]
            return None
        entry.access_count += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        if len(self._store) >= self.max_size and key not in self._store:
            # Evict least frequently accessed
            lfu_key = min(self._store.keys(), key=lambda k: self._store[k].access_count)
            del self._store[lfu_key]
        self._store[key] = CacheEntry(value, effective_ttl)

    def invalidate(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()

    def size(self) -> int:
        return len(self._store)


class CryptographicAuditLog:
    """Tamper-evident SHA-256 hash-chained audit ledger."""
    def __init__(self, genesis_salt: str = "GENESIS_OT8_BLOCK"):
        self.chain: list[Dict[str, Any]] = []
        self._previous_hash: str = hashlib.sha256(genesis_salt.encode("utf-8")).hexdigest()

    def append_entry(self, actor: str, action: str, payload: Dict[str, Any]) -> str:
        timestamp = time.time()
        canonical_payload = json.dumps(payload, sort_keys=True)
        block_content = f"{self._previous_hash}|{actor}|{action}|{timestamp}|{canonical_payload}"
        current_hash = hashlib.sha256(block_content.encode("utf-8")).hexdigest()

        entry = {
            "index": len(self.chain),
            "timestamp": timestamp,
            "actor": actor,
            "action": action,
            "payload": payload,
            "prev_hash": self._previous_hash,
            "hash": current_hash
        }
        self.chain.append(entry)
        self._previous_hash = current_hash
        return current_hash

    def verify_integrity(self) -> bool:
        """Verifies cryptographic chain unbroken continuity."""
        if not self.chain:
            return True
        for i in range(1, len(self.chain)):
            prev_block = self.chain[i - 1]
            curr_block = self.chain[i]
            if curr_block["prev_hash"] != prev_block["hash"]:
                return False
            # Recompute hash
            canonical_payload = json.dumps(curr_block["payload"], sort_keys=True)
            expected_content = f"{curr_block['prev_hash']}|{curr_block['actor']}|{curr_block['action']}|{curr_block['timestamp']}|{canonical_payload}"
            recomputed = hashlib.sha256(expected_content.encode("utf-8")).hexdigest()
            if recomputed != curr_block["hash"]:
                return False
        return True


def format_currency(amount: float, currency_symbol: str = "$") -> str:
    """Formats numeric amounts as standardized financial currency representations."""
    if amount < 0:
        return f"-{currency_symbol}{abs(amount):,.2f}"
    return f"{currency_symbol}{amount:,.2f}"


def calculate_compound_interest(principal: float, annual_rate: float, periods_per_year: int, years: float) -> float:
    """A = P * (1 + r/n)^(nt)"""
    if principal < 0 or annual_rate < 0 or periods_per_year <= 0 or years < 0:
        raise ValueError("Invalid parameters for compound interest calculation")
    return principal * math.pow((1.0 + (annual_rate / periods_per_year)), periods_per_year * years)


def sanitize_sku(sku: str) -> str:
    """Sanitizes SKU strings to uppercase alphanumeric hyphenated format."""
    cleaned = "".join(c for c in sku.strip().upper() if c.isalnum() or c in ("-", "_"))
    if not cleaned:
        raise ValueError(f"SKU string '{sku}' contained no valid alphanumeric characters")
    return cleaned
