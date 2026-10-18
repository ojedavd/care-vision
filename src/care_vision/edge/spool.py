"""Bounded encrypted samples of offline captures; no visual inference."""

import os
import sqlite3
import threading
import time
from base64 import b64decode, b64encode
from pathlib import Path
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from care_vision.edge.buffer import Capture
from care_vision.edge.sampler import Sampler


class EncryptedSpool:
    def __init__(
        self,
        path,
        key: bytes,
        *,
        max_seconds=60,
        max_frames=120,
        max_bytes=64 * 1024 * 1024,
    ):
        if len(key) != 32:
            raise ValueError("spool key must be 32 bytes")
        if not 0 < max_seconds <= 3600 or max_frames <= 0 or max_bytes <= 0:
            raise ValueError("invalid spool limits")
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("spool path must not be a symlink")
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path, 0o700)
        database = self.path / "offline.sqlite3"
        if database.is_symlink():
            raise ValueError("spool database must not be a symlink")
        descriptor = os.open(database, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(database, 0o600)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(database, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA secure_delete=ON")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS frames ("
            "seq INTEGER PRIMARY KEY, mono_ns INTEGER NOT NULL, at_utc TEXT NOT NULL, "
            "nonce BLOB NOT NULL, encrypted BLOB NOT NULL)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS frames_mono ON frames(mono_ns)")
        row = self.conn.execute("SELECT value FROM meta WHERE key='boot_id'").fetchone()
        self.boot_id = row[0] if row else str(uuid4())
        if not row:
            self.conn.execute("INSERT INTO meta(key,value) VALUES('boot_id',?)", (self.boot_id,))
        row = self.conn.execute("SELECT value FROM meta WHERE key='last_seq'").fetchone()
        self.last_seq = int(row[0]) if row else 0
        row = self.conn.execute("SELECT value FROM meta WHERE key='last_mono_ns'").fetchone()
        if row and int(row[0]) > time.monotonic_ns():
            self.conn.close()
            raise RuntimeError("monotonic clock reset; archived capture timeline is incompatible")
        self.aes = AESGCM(key)
        row = self.conn.execute("SELECT value FROM meta WHERE key='key_check'").fetchone()
        if row:
            check = b64decode(row[0])
            try:
                plaintext = self.aes.decrypt(check[:12], check[12:], self.boot_id.encode())
            except InvalidTag as exc:
                self.conn.close()
                raise ValueError("spool key does not match existing archive") from exc
            if plaintext != b"CARE-Vision spool key":
                self.conn.close()
                raise ValueError("invalid spool key check")
        else:
            nonce = os.urandom(12)
            check = nonce + self.aes.encrypt(nonce, b"CARE-Vision spool key", self.boot_id.encode())
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('key_check',?)",
                (b64encode(check).decode(),),
            )
        self.conn.commit()
        self.sampler = Sampler(2)
        self.max_seconds = max_seconds
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.evicted = 0
        self.closed = False
        self._prune(time.monotonic_ns())

    def _aad(self, seq, mono_ns, at_utc):
        return f"{self.boot_id}|{seq}|{mono_ns}|{at_utc}".encode()

    def append(self, capture: Capture, retain: bool):
        with self.lock:
            if capture.seq <= self.last_seq:
                raise ValueError("spool capture sequence must increase")
            selected = retain and self.sampler.select(capture.seq, capture.mono_ns)
            with self.conn:
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('last_seq',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(capture.seq),),
                )
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('last_mono_ns',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(capture.mono_ns),),
                )
                if selected:
                    nonce = os.urandom(12)
                    encrypted = self.aes.encrypt(
                        nonce,
                        capture.jpeg,
                        self._aad(capture.seq, capture.mono_ns, capture.at_utc),
                    )
                    self.conn.execute(
                        "INSERT INTO frames VALUES(?,?,?,?,?)",
                        (capture.seq, capture.mono_ns, capture.at_utc, nonce, encrypted),
                    )
            self.last_seq = capture.seq
            if selected or capture.mono_ns - self.last_prune_ns >= 1_000_000_000:
                self._prune(capture.mono_ns)
            return bool(selected)

    def _prune(self, now_ns):
        with self.lock:
            with self.conn:
                cursor = self.conn.execute(
                    "DELETE FROM frames WHERE mono_ns <= ?",
                    (now_ns - int(self.max_seconds * 1e9),),
                )
                self.evicted += cursor.rowcount
                while True:
                    count, size = self.conn.execute(
                        "SELECT COUNT(*), COALESCE(SUM(LENGTH(encrypted)),0) FROM frames"
                    ).fetchone()
                    if count <= self.max_frames and size <= self.max_bytes:
                        break
                    self.conn.execute("DELETE FROM frames WHERE seq=(SELECT MIN(seq) FROM frames)")
                    self.evicted += 1
            self.last_prune_ns = now_ns

    def select(self, start_ns, end_ns, fps):
        with self.lock:
            rows = self.conn.execute(
                "SELECT seq,mono_ns,at_utc,nonce,encrypted FROM frames "
                "WHERE mono_ns>=? AND mono_ns<? ORDER BY mono_ns",
                (start_ns, end_ns),
            ).fetchall()
            sampler = Sampler(fps)
            sampler.set_rate(fps, start_ns)
            return tuple(
                Capture(
                    seq,
                    mono_ns,
                    at_utc,
                    self.aes.decrypt(nonce, encrypted, self._aad(seq, mono_ns, at_utc)),
                )
                for seq, mono_ns, at_utc, nonce, encrypted in rows
                if sampler.select(seq, mono_ns)
            )

    def status(self):
        with self.lock:
            count, size, oldest, newest = self.conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(encrypted)),0), "
                "MIN(mono_ns), MAX(mono_ns) FROM frames"
            ).fetchone()
            return {
                "spool_frames": count,
                "spool_encrypted_bytes": size,
                "spool_evicted": self.evicted,
                "spool_oldest_mono_ns": oldest,
                "spool_newest_mono_ns": newest,
            }

    def acknowledge(self, seqs):
        """Discard only captures confirmed processed by a completed cloud session."""
        with self.lock:
            seqs = tuple(set(seqs))
            if not seqs:
                return 0
            with self.conn:
                cursor = self.conn.executemany(
                    "DELETE FROM frames WHERE seq=?", ((seq,) for seq in seqs)
                )
            return cursor.rowcount

    def close(self):
        with self.lock:
            if not self.closed:
                self._prune(time.monotonic_ns())
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self.conn.close()
                self.closed = True
