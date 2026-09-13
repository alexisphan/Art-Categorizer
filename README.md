# WikiArt Multi-Task Pipeline

Pinned to the **[WikiArt dataset by steubk on Kaggle](https://www.kaggle.com/datasets/steubk/wikiart)**
(~81k images, the same underlying archive as `huggan/wikiart` / the ArtGAN
"new WikiArt dataset"). Three tasks on one shared pipeline:

1. **Style classification** (27 style/movement classes)
2. **Artist attribution** (129 artist classes, including "Unknown Artist")
3. **Genre classification** (~11 genre classes, including "Unknown Genre")

> **Note on era/period classification**: the original version of this
> pipeline included an era task, but the steubk/wikiart dataset has **no
> date/year field anywhere** — not in the images, not in the provided CSVs.
> So era classification isn't trainable from this dataset alone, and
> **genre replaces era** as the third task by default. If you separately
> obtain a metadata source with completion dates (e.g. a scrape of
> wikiart.org's own per-artwork pages) and merge it in, the `era` task
> and `ERA_BIN_SIZE`/`ERA_MIN_YEAR`/`ERA_MAX_YEAR` settings in `config.py`
> are still there and will work once `year` is populated.

Downloaded from Kaggle, this dataset unpacks to:

```
<Style_1>/<artist-name_title-of-work>.jpg
<Style_2>/...
...
classes.csv     # filename, artist, genre, description, phash, width, height, genre_count, subset (no style column)
wclasses.csv    # file, artist, genre, style   <- this is the one data_prep.py uses
```

Confirmed via `--inspect` on the actual download: images are folder-per-style,
2 levels deep, and neither CSV has a date/year column, so the "no dates in
this dataset" note above is confirmed, not just anticipated. When both CSVs
are present, `data_prep.py` scores each by how many of
path/artist/style/genre it covers and picks `wclasses.csv` automatically
(since `classes.csv` is missing `style` entirely) — you don't need to name
it explicitly.

## Step 0 — Inspect your data first

Before anything else, run:

```bash
python -m src.data_prep --inspect --root /path/to/your/wikiart
```

This prints:
- Top-level folder structure (a few levels deep)
- Any CSV files found, plus their columns (or a sample row, if headerless)
- Any `*_class.txt`/`*_class.csv` class-index-to-name lookup files found
- Whether `style_train.csv`/`artist_train.csv`/`genre_train.csv`-style
  per-task CSVs were detected (the steubk/wikiart layout)
- A sample of image filenames and path depth

**Paste that output back to Claude if the pipeline errors out.**
`data_prep.py` tries three layouts, in this priority order:

1. **steubk/wikiart-style per-task CSVs** (a variant seen on some mirrors of
   this archive): `<task>_train.csv` / `<task>_val.csv` files, each row
   `path,class_index` with no header, plus a `<task>_class.txt`-style lookup
   mapping index to class name.
2. **A single unified metadata CSV** — this is what the actual
   steubk/wikiart download uses (`wclasses.csv`, with `classes.csv` as a
   sibling that's missing the style column). When more than one such CSV
   exists, `data_prep.py` picks whichever one covers the most of
   path/artist/style/genre rather than an arbitrary one.
3. **Folder-per-style only** (e.g. the ipythonx Kaggle mirror, no CSV at
   all): images live in `root/<Style>/<artist-name_title-of-work>.jpg`,
   artist is parsed from the filename, and genre/year aren't recoverable.

`data_prep.py` builds a unified `metadata.csv` with columns:
`path, artist, style, genre, year, era_label`.
Missing fields become `NaN` and that task is simply skipped/reported.

## Setup

```bash
pip install -r requirements.txt
```

GPU strongly recommended for training (WikiArt is ~80k+ images). CPU works
for the small subset / pipeline sanity-check, not for real training.

## Usage

```bash
# 1. Inspect
python -m src.data_prep --inspect --root /path/to/wikiart

# 2. Build unified metadata.csv (adjust flags based on what --inspect showed)
python -m src.data_prep --root /path/to/wikiart --out metadata.csv

# 3. Train one task at a time
python -m src.train --task style  --metadata metadata.csv --epochs 10
python -m src.train --task artist --metadata metadata.csv --epochs 10
python -m src.train --task genre  --metadata metadata.csv --epochs 10
# python -m src.train --task era  --metadata metadata.csv --epochs 10   # only if you've merged in dates

# 4. Evaluate + confusion matrix
python -m src.evaluate --task style --metadata metadata.csv --checkpoint checkpoints/style_best.pt
```

## Design decisions / assumptions (change these if wrong)

- **Backbone**: pretrained ResNet50 (ImageNet weights), fine-tuned. Swap in
  `src/model.py` if you'd rather use EfficientNet or a ViT.
- **Artist attribution**: artists with fewer than `MIN_SAMPLES_PER_ARTIST`
  (default 10) images are dropped — otherwise the classifier has classes
  with 1-2 examples, which is not learnable and wrecks stratified splitting.
- **Genre classification**: same rare-class filtering, via
  `MIN_SAMPLES_PER_GENRE` (default 10).
- **Era task**: not trainable on steubk/wikiart as downloaded (no date
  field). Left in the codebase for anyone who merges in a dated metadata
  source; dates would be binned into 50-year buckets by default
  (`ERA_BIN_SIZE` in `config.py`).
- **Splits**: 70/15/15 train/val/test, stratified by the target label per task.
- **Class imbalance**: handled via inverse-frequency class weights in the
  loss, not oversampling — simpler and works fine at this scale.
- **Image size**: 224x224, standard ImageNet normalization.

## Visualizing results in Tableau

Tableau can't run the PyTorch code directly, but `train.py` and `evaluate.py`
now export CSVs specifically for this:

| File | Produced by | Good for |
|---|---|---|
| `metadata.csv` | `data_prep.py` | Dataset composition: count of paintings per style/artist/era, bar/tree maps |
| `logs/<task>_training_history.csv` | `train.py` | Train/val loss & accuracy curves per epoch (line chart, x=epoch) |
| `logs/<task>_predictions.csv` | `evaluate.py` | Per-image results: true label, predicted label, confidence, correct/incorrect — filter/drill into misclassifications, confidence histograms, accuracy by class |
| `logs/<task>_confusion_matrix.csv` | `evaluate.py` | Long-format (true_label, predicted_label, count) — plug straight into a Tableau heatmap: columns=predicted_label, rows=true_label, color=count |

Quick Tableau recipes:
- **Confusion matrix heatmap**: connect to `*_confusion_matrix.csv` →
  Columns: `predicted_label`, Rows: `true_label`, Marks: Square, Color: `count`.
- **Training curves**: connect to `*_training_history.csv` → Columns: `epoch`,
  Rows: `train_acc` and `val_acc` as a dual-axis line chart.
- **Per-class accuracy**: connect to `*_predictions.csv` → group by
  `true_label`, calculated field `AVG(correct)` as a bar chart, sorted
  descending — the fastest way to spot which styles/artists the model
  struggles with.
- **Dataset imbalance**: connect to `metadata.csv` → COUNT of rows grouped
  by `artist` or `style` — useful as a companion chart explaining *why*
  certain classes perform worse (small sample size correlates with lower
  per-class accuracy, worth showing side-by-side).

All four CSVs share label columns (`style`, `artist`, `true_label`, etc.),
so you can also join `predictions.csv` back to `metadata.csv` in Tableau if
you want additional fields (e.g. filtering misclassifications by style even
when the task being evaluated is artist attribution).

## Known gaps to fix once you see real data

- If your WikiArt copy has no date metadata at all, the era task needs a
  different data source merged in — tell Claude and we'll add a merge step.
- If filenames don't encode artist name, `data_prep.py`'s filename-parsing
  fallback won't work and we'll need the folder structure or a CSV instead.