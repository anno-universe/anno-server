import hashlib
import os
import time
from pathlib import Path

from django.core.cache.backends.base import BaseCache


class TempFileCache(BaseCache):
    def __init__(self, location, params):
        super().__init__(params)
        self._dir = Path(location)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key):
        safe = hashlib.md5(key.encode()).hexdigest()
        return self._dir / f"{safe}.jpg"

    def filepath(self, key):
        """Return the file path for a key. Public so callers can build
        redirect headers without knowing internal hash logic."""
        return self._key_path(self.make_key(key))

    def get(self, key, default=None, version=None):
        key = self.make_key(key, version)
        path = self._key_path(key)
        if not path.exists():
            return default
        timeout = self.get_backend_timeout()
        if timeout and time.time() - path.stat().st_mtime > timeout:
            path.unlink(missing_ok=True)
            return default
        # Touch atime so OS-level cleanup (find -atime) works naturally
        os.utime(path, (time.time(), path.stat().st_mtime))
        return path.read_bytes()

    def set(self, key, value, timeout=None, version=None):
        key = self.make_key(key, version)
        path = self._key_path(key)
        path.write_bytes(value)
        # Track mtime for cache expiry
        os.utime(path, (time.time(), time.time()))

    def delete(self, key, version=None):
        key = self.make_key(key, version)
        self._key_path(key).unlink(missing_ok=True)

    def has_key(self, key, version=None):
        key = self.make_key(key, version)
        return self._key_path(key).exists()

    def clear(self):
        for f in self._dir.iterdir():
            f.unlink()
