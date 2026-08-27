"""Train flow models and cache them for the figure scripts.

Usage:
    python plots/make_cache.py moons:realnvp moons:maf      # specific pairs
    python plots/make_cache.py moons:all                    # all flows on a dataset
    python plots/make_cache.py gallery                       # the gallery grid

Each pair is trained full-batch and cached to plots/_cache/. Kept small so a
handful of pairs train within a single short run.
"""

import sys
import time

import torch

import _common as C

GALLERY_FLOWS = ["realnvp", "spline", "maf"]
ALL_FLOWS = ["realnvp", "spline", "maf", "iaf", "cnf"]
CACHE_SEED = 0


def expand(tokens):
    pairs = []
    for tok in tokens:
        if tok == "gallery":
            for ds in C.DATASETS:
                for fl in GALLERY_FLOWS:
                    pairs.append((ds, fl))
            continue
        ds, fl = tok.split(":")
        flows = ALL_FLOWS if fl == "all" else GALLERY_FLOWS if fl == "gallery" else [fl]
        for f in flows:
            pairs.append((ds, f))
    # de-dupe preserving order
    seen, out = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main(tokens):
    pairs = expand(tokens)
    for ds, fl in pairs:
        torch.manual_seed(CACHE_SEED)
        data = C.get_dataset(ds, n=C.NDATA.get(fl, 2000), seed=0)
        train_data, validation_data = C.train_validation_split(
            data, validation_fraction=0.2, seed=CACHE_SEED
        )
        model = C.build_model(fl)
        epochs = C.EPOCHS[fl]
        t = time.time()
        training_result = C.train(
            model,
            train_data,
            validation_data,
            epochs=epochs,
            lr=C.LR.get(fl, 1e-3),
        )
        dt = time.time() - t
        quality = C.validate_cache_candidate(model, validation_data, training_result)
        C.save_cache(
            ds,
            fl,
            model,
            training_result,
            quality,
            dt,
            seed=CACHE_SEED,
        )
        print(f"cached {ds:13s} {fl:8s} epochs={epochs:5d} "
              f"time={dt:5.1f}s val_nll={quality['candidate_validation_nll']:7.3f} "
              f"best_epoch={training_result['best_epoch'] + 1:4d} "
              f"params={C.count_params(model)}")


if __name__ == "__main__":
    toks = sys.argv[1:] or ["moons:gallery"]
    main(toks)
