import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitnet.runtime import TelemetryEvent
from bitnet.telemetry import TelemetryRecord, build_cli_telemetry_sink


class TelemetryWriterTests(unittest.TestCase):
    def test_writer_emits_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "telemetry.jsonl"
            writer = build_cli_telemetry_sink(output_path=path, component="test")
            event = TelemetryEvent(name="runtime.execute.start", timestamp=1.0, attributes={"ok": True})
            writer(event)
            writer.close()

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 1)
            payload = json.loads(lines[-1])
            self.assertEqual(payload["component"], "test")
            self.assertEqual(payload["event"], "runtime.execute.start")

    def test_record_round_trip(self):
        record = TelemetryRecord(timestamp=2.0, component="comp", event="evt", attributes={"a": 1})
        payload = json.dumps(record.as_dict())
        decoded = json.loads(payload)
        self.assertEqual(decoded["component"], "comp")
        self.assertEqual(decoded["event"], "evt")


if __name__ == "__main__":
    unittest.main()
