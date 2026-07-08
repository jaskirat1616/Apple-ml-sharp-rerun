"""Tests for Splatline v2 configuration and backend selection."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class SplatlineConfigTests(unittest.TestCase):
    def test_default_config(self):
        from utils.splatline_config import SplatlineConfig, TIER_SKELETON
        cfg = SplatlineConfig()
        self.assertEqual(cfg.splat_backend, "sharp")
        self.assertEqual(cfg.human_tier, TIER_SKELETON)
        self.assertFalse(cfg.first_run_complete)

    def test_config_roundtrip(self):
        from utils.splatline_config import SplatlineConfig
        cfg = SplatlineConfig(splat_backend="triposplat", human_tier="mesh", device="cuda")
        d = cfg.to_dict()
        self.assertEqual(d["splat_backend"], "triposplat")
        self.assertEqual(d["human_tier"], "mesh")
        cfg2 = SplatlineConfig.from_dict(d)
        self.assertEqual(cfg2.splat_backend, "triposplat")
        self.assertEqual(cfg2.human_tier, "mesh")

    def test_config_from_dict_ignores_unknown_keys(self):
        from utils.splatline_config import SplatlineConfig
        cfg = SplatlineConfig.from_dict({"splat_backend": "sharp", "unknown_key": "value"})
        self.assertEqual(cfg.splat_backend, "sharp")

    @patch("utils.splatline_config.CONFIG_DIR")
    @patch("utils.splatline_config.CONFIG_FILE")
    def test_save_and_load_config(self, mock_file, mock_dir):
        from utils.splatline_config import SplatlineConfig, load_config, save_config
        import utils.splatline_config as sc

        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / ".splatline"
            config_file = config_dir / "config.json"
            mock_dir.__bool__ = lambda self: True
            mock_dir.mkdir = lambda *a, **k: config_dir.mkdir(*a, **k)
            mock_file.write_text = lambda data, encoding=None: config_file.write_text(data, encoding=encoding)
            mock_file.exists = lambda self: config_file.exists()

            # Save
            cfg = SplatlineConfig(splat_backend="triposplat", human_tier="both")
            sc.CONFIG_DIR = config_dir
            sc.CONFIG_FILE = config_file
            save_config(cfg)
            self.assertTrue(config_file.exists())

            # Load
            loaded = load_config()
            self.assertEqual(loaded.splat_backend, "triposplat")
            self.assertEqual(loaded.human_tier, "both")

    def test_backend_choices_have_required_fields(self):
        from utils.splatline_config import BACKEND_CHOICES
        for choice in BACKEND_CHOICES:
            self.assertIn("key", choice)
            self.assertIn("name", choice)
            self.assertIn("license", choice)
            self.assertIn("commercial_ok", choice)

    def test_tier_choices_have_required_fields(self):
        from utils.splatline_config import HUMAN_TIER_CHOICES
        for choice in HUMAN_TIER_CHOICES:
            self.assertIn("key", choice)
            self.assertIn("name", choice)
            self.assertIn("description", choice)

    def test_sharp_is_in_choices(self):
        from utils.splatline_config import BACKEND_CHOICES
        keys = [c["key"] for c in BACKEND_CHOICES]
        self.assertIn("sharp", keys)

    def test_triposplat_is_in_choices(self):
        from utils.splatline_config import BACKEND_CHOICES
        keys = [c["key"] for c in BACKEND_CHOICES]
        self.assertIn("triposplat", keys)

    def test_vggt_is_in_choices(self):
        from utils.splatline_config import BACKEND_CHOICES
        keys = [c["key"] for c in BACKEND_CHOICES]
        self.assertIn("vggt", keys)

    def test_five_backend_choices(self):
        from utils.splatline_config import BACKEND_CHOICES
        self.assertEqual(len(BACKEND_CHOICES), 5)


if __name__ == "__main__":
    unittest.main()
