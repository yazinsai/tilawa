"""Low-compute regression tests for preserving an inherited listener bundle."""
import json
import tempfile
import unittest
from pathlib import Path

from student_acoustic_span_bundle import ASSETS, DESCRIPTOR, assemble, fingerprint, resolve_listener, snapshot


class InheritedBundle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / 'models'
        self.model = self.models / 'inherited-listener'
        self.model.mkdir(parents=True)
        (self.model / 'primary.onnx').write_bytes(b'inherited-primary')
        self.manifest = dict(file='primary.onnx', tokens='بت', inventory='test',
                             sha256=fingerprint(self.model / 'primary.onnx')['sha256'])
        (self.model / 'manifest.json').write_text(json.dumps(self.manifest))
        (self.model / 'teacher.onnx').write_bytes(b'inherited-teacher')
        (self.model / 'teacher.onnx.data').write_bytes(b'inherited-external-data')
        (self.model / 'other').mkdir()
        (self.model / 'other' / 'calibration.json').write_bytes(b'inherited-auxiliary')
        self.card = self.root / 'card.json'
        self.card.write_text(json.dumps(dict(listener=self.model.name)))
        self.fitted = self.root / 'fitted'
        self.fitted.mkdir()
        self.parameters = dict(primarySha256=self.manifest['sha256'], tokens='بت', inventory='test')
        (self.fitted / 'acoustic-span-features-metadata.json').write_text(json.dumps(self.parameters))
        for asset in ASSETS:
            (self.fitted / asset).write_bytes(b'new-head')
        (self.fitted / ASSETS[0]).write_text(json.dumps(self.parameters))
        self.update_descriptor()
        self.out = self.root / 'bundle'

    def update_descriptor(self):
        (self.fitted / DESCRIPTOR).write_text(json.dumps({file: fingerprint(self.fitted / file) for file in ASSETS}))

    def test_resolves_card_listener_and_preserves_every_inherited_asset(self):
        # A shipped bundle is present but must not be chosen over the card.
        shipped = self.models / 'spliced-clean-int8'
        shipped.mkdir()
        (shipped / 'model.onnx').write_bytes(b'wrong-primary')
        card_bytes = self.card.read_bytes()
        before = snapshot(self.model)
        self.assertEqual(resolve_listener(self.card, self.models), self.model)
        report = assemble(self.model, self.fitted, self.out)
        self.assertEqual(report['inherited'], before)
        self.assertEqual(snapshot(self.model), before)
        self.assertEqual(self.card.read_bytes(), card_bytes)
        for file in before:
            self.assertEqual((self.out / file).read_bytes(), (self.model / file).read_bytes())
        self.assertEqual(set(snapshot(self.out)), set(before) | set(ASSETS) | {DESCRIPTOR})

    def test_missing_inherited_listener_never_falls_back(self):
        self.card.write_text(json.dumps(dict(listener='unavailable-listener')))
        (self.models / 'spliced-clean-int8').mkdir()
        with self.assertRaisesRegex(FileNotFoundError, 'Inherited listener bundle is unavailable'):
            resolve_listener(self.card, self.models)

    def test_wrong_primary_in_either_feature_metadata_or_head_rejects_assembly(self):
        for file in ('acoustic-span-features-metadata.json', ASSETS[0]):
            with self.subTest(file=file):
                path = self.fitted / file
                original = path.read_bytes()
                path.write_text(json.dumps({**self.parameters, 'primarySha256': '0' * 64}))
                self.update_descriptor()
                with self.assertRaisesRegex(ValueError, 'differ from the inherited primary'):
                    assemble(self.model, self.fitted, self.out)
                self.assertFalse(self.out.exists())
                path.write_bytes(original)
                self.update_descriptor()

    def test_missing_or_corrupt_control_rejects_before_copying(self):
        control = self.fitted / 'acoustic-span-control.json'
        control.write_bytes(b'changed-control')
        with self.assertRaisesRegex(ValueError, 'Fitted asset checksum mismatch'):
            assemble(self.model, self.fitted, self.out)
        self.assertFalse(self.out.exists())
        control.unlink()
        with self.assertRaises(FileNotFoundError):
            assemble(self.model, self.fitted, self.out)

    def test_stale_output_and_inherited_asset_collisions_cannot_be_overwritten(self):
        self.out.mkdir()
        (self.out / 'stale.onnx').write_bytes(b'stale')
        with self.assertRaisesRegex(ValueError, 'fresh and empty'):
            assemble(self.model, self.fitted, self.out)
        (self.out / 'stale.onnx').unlink()
        (self.model / ASSETS[0]).write_bytes(b'inherited-duration-head')
        with self.assertRaisesRegex(ValueError, 'overwrite inherited assets'):
            assemble(self.model, self.fitted, self.out)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_corrupt_inherited_primary_cannot_be_used_for_extraction(self):
        (self.model / 'primary.onnx').write_bytes(b'changed-primary')
        with self.assertRaisesRegex(ValueError, 'Inherited primary checksum mismatch'):
            resolve_listener(self.card, self.models)


if __name__ == '__main__':
    unittest.main()
