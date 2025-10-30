import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_inference import _build_parser

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except Exception as exc:  # pragma: no cover - skip when hypothesis unavailable
    raise unittest.SkipTest(f"hypothesis not available: {exc}")


def _thread_strategy():
    return st.one_of(st.just("auto"), st.integers(min_value=1, max_value=64).map(str))


class CliPropertyTests(unittest.TestCase):
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        prompt=st.text(min_size=1, max_size=32),
        n_predict=st.integers(min_value=1, max_value=1024),
        ctx_size=st.integers(min_value=1, max_value=4096),
        temperature=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
        threads=_thread_strategy(),
        batch=st.integers(min_value=1, max_value=8),
    )
    def test_round_trip_argument_parsing(self, prompt, n_predict, ctx_size, temperature, threads, batch):
        parser = _build_parser()
        argv = [
            "--model",
            "model.gguf",
            "--prompt",
            prompt,
            "--n-predict",
            str(n_predict),
            "--ctx-size",
            str(ctx_size),
            "--temperature",
            f"{temperature:.3f}",
            "--threads",
            threads,
            "--batch-size",
            str(batch),
        ]
        args = parser.parse_args(argv)
        self.assertEqual(args.prompt, prompt)
        self.assertEqual(args.n_predict, n_predict)
        self.assertEqual(args.ctx_size, ctx_size)
        self.assertTrue(math.isclose(args.temperature, round(temperature, 3)))
        if threads == "auto":
            self.assertIsNone(args.threads)
        else:
            self.assertEqual(args.threads, int(threads))
        self.assertEqual(args.batch_size, batch)

    @settings(max_examples=20)
    @given(value=st.text().filter(lambda s: s not in {"auto"} and not s.isdigit()))
    def test_thread_parser_rejects_invalid_values(self, value):
        with self.assertRaises(SystemExit):
            _build_parser().parse_args(["--model", "m.gguf", "--prompt", "hi", "--threads", value])


class ConfigurationFuzzTests(unittest.TestCase):
    @settings(max_examples=30)
    @given(
        temperature=st.floats(min_value=-5.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        n_predict=st.integers(max_value=0),
    )
    def test_invalid_numeric_parameters_trigger_errors(self, temperature, n_predict):
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--model",
                "m.gguf",
                "--prompt",
                "p",
                "--temperature",
                str(temperature),
                "--n-predict",
                str(n_predict),
            ])


if __name__ == "__main__":
    unittest.main()
