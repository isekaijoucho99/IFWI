"""CPU/stdlib tests using fresh workspace fixtures, never a training/PID query."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE))


class SuiteControlTests(unittest.TestCase):
    def setUp(self):
        self.fixture_parent = WORKTREE / "verification" / ".suite_control_fixtures"
        self.fixture_parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.fixture_parent, prefix="test_")
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(WORKTREE))
        self.output = self.root / "output"
        self.output.mkdir()
        self.lock = self.root / "suite.lock"
        self.controls = []

    def tearDown(self):
        # Close only fds created in these harmless fixtures. Production release
        # must preserve uncertain locks; fixture destruction is not a release.
        for control in self.controls:
            fd = getattr(control, "_lock_fd", None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                control._lock_fd = None
        self.assertTrue(self.root.is_relative_to(WORKTREE))
        self.temp.cleanup()

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("ifwi_research.suite_control"),
                             "SuiteControl implementation is missing")
        return importlib.import_module("ifwi_research.suite_control")

    def control(self, output=None, run_id="test-run"):
        control = self.module().SuiteControl(output or self.output, run_id, self.lock)
        self.controls.append(control)
        return control

    def load(self, relative):
        return json.loads((self.output / relative).read_text(encoding="utf-8"))

    def write_stop(self, **payload):
        (self.output / "STOP_REQUEST.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_acquire_records_actual_identity_and_exclusive_lock(self):
        control = self.control()
        launch = control.acquire()
        lock = json.loads(self.lock.read_text(encoding="utf-8"))
        self.assertEqual(launch, self.load("control/LAUNCH.json"))
        self.assertEqual(lock["run_id"], "test-run")
        self.assertEqual(lock["owner_pid"], os.getpid())
        self.assertEqual(launch["pid"], os.getpid())
        self.assertEqual(launch["parent_pid"], os.getppid())
        self.assertIn("created_utc", launch)
        other = self.root / "other-output"
        other.mkdir()
        before = self.lock.read_bytes()
        with self.assertRaises(FileExistsError):
            self.control(other, "other-run").acquire()
        self.assertEqual(before, self.lock.read_bytes())
        self.assertFalse((other / "control" / "LAUNCH.json").exists())

    def test_stale_lock_is_not_automatically_removed(self):
        self.lock.write_text('{"run_id":"old","owner_pid":123}', encoding="utf-8")
        before = self.lock.read_bytes()
        with self.assertRaises(FileExistsError):
            self.control().acquire()
        self.assertEqual(self.lock.read_bytes(), before)

    def test_existing_control_directory_is_not_reused(self):
        (self.output / "control").mkdir()
        marker = self.output / "control" / "old-evidence.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.control().acquire()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse(self.lock.exists())

    def test_missing_stop_is_noop(self):
        control = self.control()
        control.acquire()
        self.assertIsNone(control.check_stop())
        self.assertFalse((self.output / "control" / "STOP_ACK.json").exists())

    def test_matching_stop_is_acknowledged_only_after_terminal_receipt(self):
        control = self.control()
        control.acquire()
        self.write_stop(run_id="test-run", action="stop", reason="fixture request")
        with self.assertRaises(self.module().StopRequested):
            control.check_stop()
        self.assertFalse((self.output / "control" / "STOP_ACK.json").exists())
        receipt = control.finish("STOPPED_BY_USER", 130, checkpoint="safe.pt", update=7)
        self.assertEqual(receipt["status"], "STOPPED_BY_USER")
        self.assertNotEqual(receipt["status"], "COMPLETED")
        ack = self.load("control/STOP_ACK.json")
        self.assertEqual(ack["run_id"], "test-run")
        self.assertEqual(ack["status"], "STOPPED_BY_USER")
        self.assertEqual(self.load("control/EXIT.json"), receipt)
        control.release()
        self.assertFalse(self.lock.exists())

    def test_wrong_stop_run_id_is_error_without_ack(self):
        control = self.control()
        control.acquire()
        self.write_stop(run_id="another-run", action="stop")
        with self.assertRaises(ValueError):
            control.check_stop()
        self.assertFalse((self.output / "control" / "STOP_ACK.json").exists())

    def test_invalid_stop_json_and_action_are_explicit_errors(self):
        control = self.control()
        control.acquire()
        path = self.output / "STOP_REQUEST.json"
        path.write_text("{bad", encoding="utf-8")
        with self.assertRaises(ValueError):
            control.check_stop()
        self.write_stop(run_id="test-run", action="pause")
        with self.assertRaises(ValueError):
            control.check_stop()
        self.assertFalse((self.output / "control" / "STOP_ACK.json").exists())

    def test_stop_terminal_status_requires_a_matching_observed_request(self):
        control = self.control()
        control.acquire()
        with self.assertRaises(RuntimeError):
            control.finish("STOPPED_BY_USER", 130)
        self.assertFalse((self.output / "control" / "STOP_ACK.json").exists())

    def test_completed_and_failed_are_distinct_and_do_not_ack_stop(self):
        control = self.control()
        control.acquire()
        receipt = control.finish("COMPLETED", 0, completed_updates=1000)
        self.assertEqual(receipt["status"], "COMPLETED")
        self.assertFalse((self.output / "control" / "STOP_ACK.json").exists())
        control.release()
        second = self.root / "failed-output"
        second.mkdir()
        failed = self.control(second, "failed-run")
        failed.acquire()
        receipt = failed.finish("FAILED", 1, error="fixture failure")
        self.assertEqual(receipt["status"], "FAILED")
        self.assertFalse((second / "control" / "STOP_ACK.json").exists())
        failed.release()

    def test_record_is_one_append_and_preserves_caller_scientific_counts(self):
        control = self.control()
        control.acquire()
        row = control.record("update_complete", phase="test", F=13, B=13, update=1)
        rows = (self.output / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0]), row)
        self.assertEqual((row["F"], row["B"], row["phase"]), (13, 13, "test"))

    def test_ledger_append_failure_is_not_retried_or_published(self):
        control = self.control()
        control.acquire()
        (self.output / "ledger.jsonl").mkdir()
        with patch.object(self.module().os, "replace", wraps=os.replace) as replace:
            with self.assertRaises(OSError):
                control.record("must_fail", F=1, B=0)
            self.assertEqual(replace.call_count, 0)

    def test_metadata_path_cannot_escape_output(self):
        control = self.control()
        control.acquire()
        for path in ("../escape.json", str(self.root / "absolute.json")):
            with self.assertRaises(ValueError):
                control.write_json(path, {"unsafe": True})
        self.assertFalse((self.root / "escape.json").exists())
        self.assertFalse((self.root / "absolute.json").exists())

    def test_nonretryable_metadata_error_is_immediate(self):
        control = self.control()
        control.acquire()
        with patch.object(self.module().os, "replace", side_effect=FileNotFoundError("fixture")) as replace:
            with self.assertRaises(FileNotFoundError):
                control.write_json("metadata.json", {"seq": 1})
            self.assertEqual(replace.call_count, 1)

    def test_only_documented_windows_metadata_errors_are_retried(self):
        control = self.control()
        control.acquire()
        real_replace = os.replace
        attempts = []
        def replace(src, dst):
            attempts.append((str(src), str(dst)))
            if len(attempts) <= 3:
                error = PermissionError("fixture sharing/access error")
                error.winerror = (5, 32, 33)[len(attempts)-1]
                raise error
            return real_replace(src, dst)
        with patch.object(self.module().os, "replace", side_effect=replace):
            control.write_json("metadata.json", {"seq": 1})
        self.assertEqual(len(attempts), 4)
        self.assertEqual(len(set(attempts)), 1)
        self.assertEqual(self.load("metadata.json"), {"seq": 1})

    @unittest.skipUnless(os.name == "nt", "Real Windows target-file sharing test")
    def test_transient_real_reader_lock_recovers_without_second_ledger_append(self):
        control = self.control()
        control.acquire()
        control.record("done_once", F=1, B=1)
        control.write_json("CURRENT.json", {"seq": 0})
        reader = (self.output / "CURRENT.json").open("r", encoding="utf-8")
        real_replace = os.replace
        attempts = []
        def replace(src, dst):
            attempts.append((str(src), str(dst)))
            try:
                return real_replace(src, dst)
            except OSError:
                reader.close()
                raise
        try:
            with patch.object(self.module().os, "replace", side_effect=replace):
                control.write_json("CURRENT.json", {"seq": 1})
        finally:
            reader.close()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(set(attempts)), 1)
        self.assertEqual(self.load("CURRENT.json"), {"seq": 1})
        self.assertEqual(len((self.output / "ledger.jsonl").read_text(encoding="utf-8").splitlines()), 1)

    @unittest.skipUnless(os.name == "nt", "Real Windows target-file sharing test")
    def test_persistent_real_reader_lock_fails_six_times_without_duplicate_append(self):
        control = self.control()
        control.acquire()
        control.record("done_once", F=1, B=1)
        ledger_before = (self.output / "ledger.jsonl").read_bytes()
        control.write_json("CURRENT.json", {"seq": 0})
        with (self.output / "CURRENT.json").open("r", encoding="utf-8"):
            with patch.object(self.module().os, "replace", wraps=os.replace) as replace:
                with self.assertRaises(self.module().MetadataPublishError) as caught:
                    control.write_json("CURRENT.json", {"seq": 1})
                self.assertEqual(replace.call_count, 6)
                self.assertEqual(caught.exception.attempts, 6)
        self.assertEqual(self.load("CURRENT.json"), {"seq": 0})
        self.assertEqual((self.output / "ledger.jsonl").read_bytes(), ledger_before)

    def test_release_requires_written_terminal_receipt(self):
        control = self.control()
        control.acquire()
        before = self.lock.read_bytes()
        with self.assertRaises(RuntimeError):
            control.release()
        self.assertEqual(self.lock.read_bytes(), before)

    def test_release_does_not_remove_another_run_or_owner_lock(self):
        control = self.control()
        control.acquire()
        own = self.lock.read_bytes()
        control.finish("FAILED", 1)
        for changes in ({"run_id": "another-run"}, {"owner_pid": os.getpid()+1}):
            row = json.loads(own)
            row.update(changes)
            self.lock.write_text(json.dumps(row), encoding="utf-8")
            before = self.lock.read_bytes()
            with self.assertRaises(RuntimeError):
                control.release()
            self.assertEqual(self.lock.read_bytes(), before)
        self.lock.write_bytes(own)
        control.release()
        self.assertFalse(self.lock.exists())

    def test_release_checks_actual_fd_inode_before_removing_lock(self):
        control = self.control()
        control.acquire()
        control.finish("FAILED", 1)
        own_fd = control._lock_fd
        alien = self.root / "alien.lock"
        alien.write_text("different inode", encoding="utf-8")
        alien_fd = os.open(alien, os.O_RDONLY)
        try:
            control._lock_fd = alien_fd
            with self.assertRaises(RuntimeError):
                control.release()
            self.assertTrue(self.lock.exists())
        finally:
            control._lock_fd = own_fd
            os.close(alien_fd)
        control.release()

    def test_changed_exit_receipt_prevents_release(self):
        control = self.control()
        control.acquire()
        control.finish("COMPLETED", 0)
        path = self.output / "control" / "EXIT.json"
        own = path.read_bytes()
        row = json.loads(own)
        row["run_id"] = "other-run"
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            control.release()
        self.assertTrue(self.lock.exists())
        path.write_bytes(own)
        control.release()

    def test_source_has_no_other_process_query_interfaces(self):
        self.module()
        source = (WORKTREE / "ifwi_research" / "suite_control.py").read_text(encoding="utf-8")
        for forbidden in ("import subprocess", "import psutil", "import ctypes", "OpenProcess(",
                          "Get-Process", "GetProcessById", "Win32_Process", "os.kill("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
