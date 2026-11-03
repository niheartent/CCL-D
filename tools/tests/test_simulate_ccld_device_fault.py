import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DeviceFaultSimulationTest(unittest.TestCase):
    def test_stall_generates_a_zero_progress_rank(self):
        script = Path(__file__).with_name("simulate_ccld_device_fault.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "complete.jsonl"
            subprocess.run([sys.executable, str(script), str(output), "--fault", "stall"], check=True)
            with output.open() as stream:
                rows = [json.loads(line) for line in stream]
        stalled = [row for row in rows if row["rank"] == 2]
        self.assertTrue(stalled)
        self.assertTrue(all(row["send_count"] == 0 and not row["complete"] for row in stalled))

    def test_missing_fault_omits_last_rank(self):
        script = Path(__file__).with_name("simulate_ccld_device_fault.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "complete.jsonl"
            subprocess.run([sys.executable, str(script), str(output), "--fault", "missing"], check=True)
            with output.open() as stream:
                rows = [json.loads(line) for line in stream]
        self.assertNotIn(3, {row["rank"] for row in rows})


if __name__ == "__main__":
    unittest.main()
