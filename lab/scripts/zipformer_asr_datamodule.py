"""Lhotse datamodule for phoneme Zipformer-CTC (drop-in for icefall zipformer).

Reads Task 3 ``/vol/manifests/<source>_cuts_fbank.jsonl.gz`` CutSets, muxes by
hours, holds out 1 % of clips by SHA-256 of cut id. Same class name as icefall's
``LibriSpeechAsrDataModule`` so ``train.py``'s import stays valid after we
overwrite this file in the copied recipe dir.

This module is imported only inside the Modal image (lhotse + icefall).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa: F401
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import AudioSamples, OnTheFlyFeatures  # noqa: F401
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


def _is_dev_id(cut_id: str, pct: int = 1) -> bool:
    h = int(hashlib.sha256(cut_id.encode("utf-8")).hexdigest(), 16)
    return (h % 100) < pct


def _cap_or_repeat(cuts: CutSet, limit: int) -> CutSet:
    items = list(cuts)
    if not items:
        return CutSet.from_cuts([])
    if len(items) >= limit:
        return CutSet.from_cuts(items[:limit])
    reps = []
    i = 0
    while len(reps) < limit:
        c = items[i % len(items)]
        reps.append(c.with_id(f"{c.id}_r{len(reps)}"))
        i += 1
    return CutSet.from_cuts(reps)


def split_train_dev(cuts: CutSet, pct: int = 1) -> tuple[CutSet, CutSet]:
    """Hold out ``pct``% of clips; guarantee ≥1 dev cut when n≥2."""
    items = list(cuts)
    if not items:
        empty = CutSet.from_cuts([])
        return empty, empty
    dev_items = [c for c in items if _is_dev_id(c.id, pct)]
    train_items = [c for c in items if not _is_dev_id(c.id, pct)]
    if not dev_items and len(items) >= 2:
        items_sorted = sorted(
            items, key=lambda c: hashlib.sha256(c.id.encode("utf-8")).hexdigest()
        )
        dev_items = [items_sorted[0]]
        train_items = items_sorted[1:]
    elif not train_items:
        train_items = list(dev_items)
    return CutSet.from_cuts(train_items), CutSet.from_cuts(dev_items)


class LibriSpeechAsrDataModule:
    """Same public surface as icefall's librispeech datamodule."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._train: Optional[CutSet] = None
        self._dev: Optional[CutSet] = None
        self._load_and_split()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="Phoneme CutSets on the zipformer-ctc-training volume.",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="Kept for icefall train.py compatibility; always our mux.",
        )
        group.add_argument(
            "--mini-libri",
            type=str2bool,
            default=False,
            help="Ignored; kept for icefall train.py compatibility.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("/vol/manifests"),
            help="Directory with <source>_cuts_fbank.jsonl.gz",
        )
        group.add_argument(
            "--sources",
            type=str,
            default="everyayah_rx,qua_rx,iqra_rx,retasy_rx,tlog_clean_v3",
            help="Comma-separated source keys to mux.",
        )
        group.add_argument(
            "--source-weights",
            type=str,
            default="",
            help="Comma-separated source=factor mux overrides (e.g. everyayah_multi=2.5). "
            "Multiplies that source's hours-weight; default 1.0.",
        )
        group.add_argument(
            "--limit-cuts",
            type=int,
            default=0,
            help="If >0, take at most this many cuts after mux (smoke).",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a batch.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="DynamicBucketingSampler vs SimpleCutSampler.",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="Buckets for DynamicBucketingSampler.",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="Concatenate utterances to reduce padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Max concatenated cut duration vs longest in batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="Padding seconds between concatenated cuts.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="On-the-fly fbank instead of precomputed features.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="Shuffle training examples each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Drop last incomplete batch.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="Include cuts on the batch.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="Training dataloader workers.",
        )
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="SpecAugment on training features.",
        )
        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="SpecAugment time-warp factor.",
        )
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=False,
            help="MUSAN noise mix; off by default (no musan cuts on vol).",
        )
        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )

    def _source_list(self) -> list[str]:
        return [s.strip() for s in str(self.args.sources).split(",") if s.strip()]

    def _load_source(self, source: str) -> Optional[CutSet]:
        path = Path(self.args.manifest_dir) / f"{source}_cuts_fbank.jsonl.gz"
        if not path.is_file():
            logging.warning("missing cuts for source %s at %s", source, path)
            return None
        logging.info("loading %s", path)
        return load_manifest_lazy(path)

    def _load_and_split(self) -> None:
        if self._train is not None:
            return
        from zipformer_ctc_utils import parse_source_weights, scale_mux_weights

        sets: list[CutSet] = []
        hours: list[float] = []
        loaded: list[str] = []
        for src in self._source_list():
            cs = self._load_source(src)
            if cs is None:
                continue
            try:
                dur = float(cs.total_duration)
            except (AttributeError, TypeError):
                dur = float(sum(c.duration for c in cs))
            if dur <= 0:
                continue
            sets.append(cs)
            hours.append(dur / 3600.0)
            loaded.append(src)
            logging.info("source %s: %.2f h", src, hours[-1])
        if not sets:
            raise FileNotFoundError(
                f"no fbank cut manifests in {self.args.manifest_dir} "
                f"for sources {self._source_list()}"
            )
        overrides = parse_source_weights(
            getattr(self.args, "source_weights", "") or ""
        )
        weights = scale_mux_weights(hours, loaded, overrides)
        for src, h, w in zip(loaded, hours, weights):
            logging.info(
                "mux %s: %.2f h × %.4g = %.2f",
                src,
                h,
                overrides.get(src, 1.0),
                w,
            )
        if len(sets) == 1:
            cuts = sets[0]
        else:
            cuts = CutSet.mux(*sets, weights=weights)
        limit = int(getattr(self.args, "limit_cuts", 0) or 0)
        if limit > 0:
            cuts = _cap_or_repeat(cuts, limit)
            logging.info("limit-cuts: %s (after cap/repeat)", limit)
        train, dev = split_train_dev(cuts, pct=1)
        logging.info("train cuts=%s dev cuts=%s", len(train), len(dev))
        self._train = train
        self._dev = dev

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )

        train = K2SpeechRecognitionDataset(
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )
        else:
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )

        if sampler_state_dict is not None:
            train_sampler.load_state_dict(sampler_state_dict)

        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)
        return DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
            )
        n_valid = len(cuts_valid)
        n_buckets = max(1, min(int(self.args.num_buckets), n_valid))
        if n_valid < 2 or n_buckets < 2:
            valid_sampler = SimpleCutSampler(
                cuts_valid,
                max_duration=self.args.max_duration,
                shuffle=False,
            )
        else:
            valid_sampler = DynamicBucketingSampler(
                cuts_valid,
                max_duration=self.args.max_duration,
                shuffle=False,
                drop_last=False,
                num_buckets=n_buckets,
            )
        return DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
        sampler = DynamicBucketingSampler(
            cuts, max_duration=self.args.max_duration, shuffle=False, drop_last=False
        )
        return DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        self._load_and_split()
        return self._train

    def train_clean_100_cuts(self) -> CutSet:
        return self.train_all_shuf_cuts()

    def train_clean_5_cuts(self) -> CutSet:
        return self.train_all_shuf_cuts()

    def train_clean_360_cuts(self) -> CutSet:
        return CutSet.from_cuts([])

    def train_other_500_cuts(self) -> CutSet:
        return CutSet.from_cuts([])

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        self._load_and_split()
        return self._dev

    def dev_other_cuts(self) -> CutSet:
        return CutSet.from_cuts([])

    def dev_clean_2_cuts(self) -> CutSet:
        return self.dev_clean_cuts()
