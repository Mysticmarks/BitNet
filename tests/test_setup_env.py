import sys
import tempfile
import unittest
from pathlib import Path

from setup_env import (
    CommandExecutionError,
    SetupConfig,
    resolve_architecture,
    run_command,
    system_info,
    validate_configuration,
)


class RunCommandTests(unittest.TestCase):
    def test_run_command_success_logs_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            result = run_command(
                [sys.executable, "-c", "print('hello world')"],
                log_step="greeting",
                log_dir=log_dir,
            )
            self.assertEqual(result.returncode, 0)
            log_path = log_dir / "greeting.log"
            self.assertTrue(log_path.exists())
            self.assertIn("hello world", log_path.read_text())

    def test_run_command_failure_raises_custom_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            with self.assertRaises(CommandExecutionError):
                run_command(
                    [sys.executable, "-c", "import sys; sys.exit(5)"],
                    log_step="failure",
                    log_dir=log_dir,
                )
            self.assertTrue((log_dir / "failure.log").exists())


class ConfigurationValidationTests(unittest.TestCase):
    def test_validate_configuration_accepts_supported_values(self):
        config = SetupConfig(
            hf_repo="tiiuae/Falcon3-3B-Instruct-1.58bit",
            model_dir="models",
            log_dir="logs",
            quant_type="i2_s",
            quant_embd=False,
            use_pretuned=False,
        )
        _, arch = system_info()
        validate_configuration(config, arch)

    def test_validate_configuration_rejects_unknown_repo(self):
        config = SetupConfig(
            hf_repo="unknown/repo",
            model_dir="models",
            log_dir="logs",
            quant_type="i2_s",
            quant_embd=False,
            use_pretuned=False,
        )
        _, arch = system_info()
        with self.assertRaises(ValueError):
            validate_configuration(config, arch)

    def test_validate_configuration_rejects_invalid_quant(self):
        _, arch = system_info()
        config = SetupConfig(
            hf_repo=None,
            model_dir="models",
            log_dir="logs",
            quant_type="unsupported_quant",
            quant_embd=False,
            use_pretuned=False,
        )
        with self.assertRaises(ValueError):
            validate_configuration(config, arch)


class ArchitectureResolutionTests(unittest.TestCase):
    def test_resolve_architecture_known_alias(self):
        self.assertEqual(resolve_architecture("x86_64"), "x86_64")

    def test_resolve_architecture_unknown_alias(self):
        with self.assertRaises(RuntimeError):
            resolve_architecture("mystery-arch")


if __name__ == "__main__":
    unittest.main()
