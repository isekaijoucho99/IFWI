"""Local suite identity, exclusive ownership, explicit stop and exit receipts.

Only metadata publication is retried. Scientific steps and ledger appends are
never retried here. The caller saves its last safe checkpoint before finish.
An existing lock is never removed automatically to make another run start.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid


class StopRequested(RuntimeError):
    """A valid stop request for this exact run was observed."""


class MetadataPublishError(RuntimeError):
    """The same prepared metadata file could not be published in six tries."""
    def __init__(self, source, target, attempts, winerror):
        self.source = str(source)
        self.target = str(target)
        self.attempts = attempts
        self.winerror = winerror
        super().__init__(f"Metadata publish failed after {attempts} attempts: "
                         f"{source} -> {target}; winerror={winerror}")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json_text(payload):
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


class SuiteControl:
    """Own one fresh output/control directory and one explicit lock file.

    acquire returns LAUNCH; record returns a flat event row; write_json returns
    its Path; finish returns EXIT. Terminal statuses are COMPLETED (code 0),
    FAILED (nonzero), or STOPPED_BY_USER after check_stop accepted a request.
    release requires all required terminal publications to have succeeded.
    """
    def __init__(self, output: Path, run_id: str, lock_path: Path):
        self.output = Path(output).resolve(strict=True)
        if not self.output.is_dir():
            raise ValueError("output must be an existing fresh directory")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        self.run_id = run_id
        self.control = self.output / "control"
        # Do not resolve an existing lock symlink and redirect O_EXCL elsewhere.
        self.lock_path = Path(lock_path).absolute()
        self._owner_pid = os.getpid()
        self._parent_pid = os.getppid()
        self._lock_fd = None
        self._lock_identity = None
        self._lock_payload = None
        self._acquired = False
        self._terminal = None
        self._stop_request = None
        self._stop_ack = None

    def _identity(self, stat):
        identity = (stat.st_dev, stat.st_ino)
        if not stat.st_ino:
            raise RuntimeError("File identity unavailable; ownership is uncertain")
        return identity

    def _running_owner(self):
        if not self._acquired or self._lock_fd is None or os.getpid() != self._owner_pid:
            raise RuntimeError("This process does not hold the suite lock")

    def _row(self, details, **fields):
        reserved = {"run_id", "pid", "owner_pid", "parent_pid", "utc"} | set(fields)
        if reserved.intersection(details):
            raise ValueError("Details cannot replace run identity or control fields")
        return dict(run_id=self.run_id, pid=self._owner_pid,
                    owner_pid=self._owner_pid, parent_pid=self._parent_pid,
                    utc=_utc(), **fields, **details)

    def acquire(self):
        if self._acquired or self._lock_fd is not None:
            raise RuntimeError("Suite lock acquisition was already attempted")
        # Both directories/files are new; never reuse old control evidence.
        self.control.mkdir(exist_ok=False)
        payload = dict(run_id=self.run_id, pid=self._owner_pid,
                       owner_pid=self._owner_pid, parent_pid=self._parent_pid,
                       created_utc=_utc())
        encoded = _json_text(payload).encode("utf-8")
        self._lock_fd = os.open(self.lock_path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0), 0o600)
        self._lock_identity = self._identity(os.fstat(self._lock_fd))
        # A write failure remains visible and leaves the new lock as evidence.
        written = os.write(self._lock_fd, encoded)
        if written != len(encoded):
            raise OSError("Incomplete suite-lock identity write; do not restart")
        os.fsync(self._lock_fd)
        self._lock_payload = payload
        self._acquired = True
        launch = dict(payload, output=str(self.output), lock_path=str(self.lock_path),
                      automatic_restart=False)
        self.write_json("control/LAUNCH.json", launch)
        return launch

    def _target(self, relative_name):
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Metadata path must stay relative to output")
        target = (self.output / relative).resolve()
        if target == self.output or not target.is_relative_to(self.output):
            raise ValueError("Metadata path escapes output")
        return target

    def write_json(self, relative_name, payload):
        """Write once, close, then retry only this same atomic replace.

        Five delays total 250ms, with six attempts. This is an attempt budget,
        not a hard wall-clock deadline. Failed temporary files remain evidence.
        """
        target = self._target(relative_name)
        text = _json_text(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        delays = (0.01, 0.02, 0.04, 0.08, 0.10)
        for index in range(6):
            try:
                os.replace(temp, target)
                return target
            except OSError as error:
                code = getattr(error, "winerror", None)
                if code not in (5, 32, 33):
                    raise
                if index == 5:
                    raise MetadataPublishError(temp, target, 6, code) from error
                time.sleep(delays[index])

    def record(self, event, **details):
        """Exactly one append; caller owns phases and all F/B accounting."""
        self._running_owner()
        if self._terminal is not None:
            raise RuntimeError("Cannot record scientific work after a terminal receipt")
        if not isinstance(event, str) or not event:
            raise ValueError("event must be nonempty")
        row = self._row(details, event=event)
        text = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        with (self.output / "ledger.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(text)
        return row

    def check_stop(self):
        """Invalid/mismatched requests fail explicitly; no acknowledgments here."""
        path = self._target("STOP_REQUEST.json")
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if (not isinstance(request, dict) or request.get("run_id") != self.run_id
                or request.get("action") != "stop"):
            raise ValueError("Invalid STOP_REQUEST action or run_id; not acknowledged")
        self._stop_request = request
        raise StopRequested(f"Stop requested for run {self.run_id}")

    def finish(self, status, exit_code, **details):
        self._running_owner()
        if self._terminal is not None:
            raise RuntimeError("A terminal receipt was already published")
        if status not in ("COMPLETED", "STOPPED_BY_USER", "FAILED"):
            raise ValueError("Unknown terminal status")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise ValueError("exit_code must be an integer")
        if (status == "COMPLETED" and exit_code != 0) or (status == "FAILED" and exit_code == 0):
            raise ValueError("Terminal status and exit_code disagree")
        if status == "STOPPED_BY_USER" and self._stop_request is None:
            raise RuntimeError("Cannot acknowledge a stop that was not accepted")
        receipt = self._row(details, status=status, exit_code=exit_code,
                            completed=(status == "COMPLETED"))
        self.write_json("control/EXIT.json", receipt)
        if status == "STOPPED_BY_USER":
            ack = dict(receipt, request=self._stop_request)
            self.write_json("control/STOP_ACK.json", ack)
            self._stop_ack = ack
        # Set only after every required publication succeeded.
        self._terminal = receipt
        return receipt

    def _validate_lock(self):
        self._running_owner()
        if (self._identity(os.fstat(self._lock_fd)) != self._lock_identity
                or self._identity(self.lock_path.lstat()) != self._lock_identity):
            raise RuntimeError("Suite lock inode changed; refusing release")
        lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
        if (not isinstance(lock, dict) or lock.get("run_id") != self.run_id
                or lock.get("owner_pid") != self._owner_pid or lock != self._lock_payload):
            raise RuntimeError("Suite lock owner changed; refusing release")

    def release(self):
        """Remove only our lock after a verified terminal publication.

        On Windows the CRT fd prevents deleting an open file, so validate the
        fd/path before closing, then recheck the path identity/content before
        unlinking. No fallback or permission workaround is attempted.
        """
        self._running_owner()
        if self._terminal is None:
            raise RuntimeError("No successfully published terminal receipt; lock retained")
        exit_row = json.loads(self._target("control/EXIT.json").read_text(encoding="utf-8"))
        if exit_row != self._terminal:
            raise RuntimeError("Terminal receipt changed; lock retained")
        if self._stop_ack is not None:
            ack = json.loads(self._target("control/STOP_ACK.json").read_text(encoding="utf-8"))
            if ack != self._stop_ack:
                raise RuntimeError("Stop acknowledgment changed; lock retained")
        self._validate_lock()
        os.close(self._lock_fd)
        self._lock_fd = None
        if self._identity(self.lock_path.lstat()) != self._lock_identity:
            raise RuntimeError("Lock path changed after close; refusing removal")
        if json.loads(self.lock_path.read_text(encoding="utf-8")) != self._lock_payload:
            raise RuntimeError("Lock owner changed after close; refusing removal")
        self.lock_path.unlink()
        self._acquired = False
