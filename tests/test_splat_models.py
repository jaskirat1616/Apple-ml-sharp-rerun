import unittest

from utils.splat_models import (
    SHARP_DEFAULT_MODEL_URL,
    get_splat_backend,
    list_splat_backends,
)


class SplatModelRegistryTests(unittest.TestCase):
    def test_sharp_backend_is_implemented_default_contract(self):
        info = get_splat_backend("sharp")
        self.assertEqual(info.status, "implemented")
        self.assertIn("PLY", info.output_contract)
        self.assertIn("single", info.input_mode)
        self.assertTrue(SHARP_DEFAULT_MODEL_URL.endswith(".pt"))

    def test_research_backends_are_tracked(self):
        keys = {backend.key for backend in list_splat_backends()}
        self.assertIn("vggt", keys)
        self.assertIn("depthsplat", keys)
        self.assertIn("noposplat", keys)
        self.assertIn("anysplat", keys)
        self.assertIn("volsplat", keys)

    def test_unknown_backend_raises_clear_error(self):
        with self.assertRaises(ValueError):
            get_splat_backend("not-a-model")

    # v2 tests
    def test_triposplat_backend_is_implemented(self):
        info = get_splat_backend("triposplat")
        self.assertEqual(info.status, "implemented")
        self.assertTrue(info.commercial_ok)
        self.assertIn("MIT", info.license)

    def test_sharp_is_non_commercial(self):
        info = get_splat_backend("sharp")
        self.assertFalse(info.commercial_ok)
        self.assertIn("non-commercial", info.license.lower())

    def test_longsplat_is_tracked(self):
        keys = {backend.key for backend in list_splat_backends()}
        self.assertIn("longsplat", keys)
        self.assertIn("splinegs", keys)

    def test_all_backends_have_license_fields(self):
        for backend in list_splat_backends():
            self.assertTrue(backend.license, f"{backend.key} missing license")
            self.assertIsInstance(backend.commercial_ok, bool)


if __name__ == "__main__":
    unittest.main()

