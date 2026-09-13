"""
Data preparation for the WikiArt pipeline.

Two jobs:
  1. --inspect: look at a dataset root and report its structure so you can
     tell Claude / decide which code path applies.
  2. (default): build a unified metadata.csv with columns
     [path, artist, style, genre, year, era_label] from whichever layout is
     detected, in priority order:
       (a) per-task class-index CSVs (e.g. style_train.csv/style_val.csv,
           artist_*, genre_*) each mapping an image path to an integer
           class index, plus optional class-name lookup files mapping
           index -> label name. (Seen on some mirrors; not the actual
           steubk/wikiart layout, see (b).)
       (b) the confirmed steubk/wikiart (Kaggle) layout: images in
           folder-per-style directories (style comes from the folder name
           — reliable and human-readable) + a "classes.csv"-style metadata
           file with real artist names and a stringified list of genre
           tags per image, joined by its filename column. This dataset
           also ships a wclasses.csv with the same fields as integer class
           codes and no shipped name lookup — we deliberately don't use
           that one, since folder name + classes.csv gives us readable
           labels for free.
       (c) a single existing CSV with recognizable column names (other
           mirrors sometimes use this).
       (d) a bare folder-per-style layout with artist encoded in the
           filename and no metadata CSV at all.

This is the file most likely to need hand-editing once you see your real
data, since WikiArt is distributed in several incompatible ways across
Kaggle/mirrors. Read the printed --inspect output carefully.

Note on steubk/wikiart specifically: it has NO date/year field anywhere,
so `year` will be entirely empty and the "era" task will not be trainable
from this dataset alone (this is expected and reported at build time —
see README "Known gaps"). That dataset does have genre labels, so "genre"
is available as a third task instead.
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Column name variants we'll try to auto-map from an existing single CSV.
COLUMN_ALIASES = {
    "artist": ["artist", "author", "painter"],
    "style": ["style", "movement"],
    "genre": ["genre", "category"],
    "year": ["date", "year", "completion_year", "creation_year"],
    "path": ["path", "filename", "file", "image", "image_path"],
}

# Tasks we know how to pull out of steubk/wikiart-style per-task CSVs.
TASK_KEYWORDS = ["style", "artist", "genre"]


def inspect(root: str, max_depth: int = 2, sample_n: int = 8) -> None:
    root = Path(root)
    if not root.exists():
        print(f"ERROR: path does not exist: {root}")
        return

    print(f"Inspecting: {root}\n")

    # 1. Top-level structure
    print("--- Top-level contents ---")
    entries = sorted(root.iterdir())
    for e in entries[:40]:
        kind = "DIR " if e.is_dir() else "file"
        print(f"  [{kind}] {e.name}")
    if len(entries) > 40:
        print(f"  ... and {len(entries) - 40} more")

    # 2. Any CSVs?
    csvs = list(root.rglob("*.csv"))
    print(f"\n--- CSV files found: {len(csvs)} ---")
    for c in csvs[:20]:
        print(f"  {c.relative_to(root)}")
        try:
            df = pd.read_csv(c, nrows=5)
            print(f"    columns: {list(df.columns)}")
        except Exception as e:
            print(f"    (couldn't read with header: {e})")
            try:
                df = pd.read_csv(c, nrows=5, header=None)
                print(f"    no header, {df.shape[1]} columns, sample row: {df.iloc[0].tolist()}")
            except Exception as e2:
                print(f"    (couldn't read at all: {e2})")

    # 3. Any plain-text class lookup files? (idx -> name, e.g. style_class.txt)
    txts = [p for p in root.rglob("*.txt") if "class" in p.name.lower()]
    if txts:
        print(f"\n--- Class-name lookup .txt files found: {len(txts)} ---")
        for t in txts[:10]:
            print(f"  {t.relative_to(root)}")
            try:
                with open(t) as f:
                    lines = [next(f) for _ in range(3)]
                print(f"    sample lines: {[l.strip() for l in lines]}")
            except Exception as e:
                print(f"    (couldn't read: {e})")

    # 4. Detected per-task CSVs (rarer alternate layout)
    task_csvs = _find_task_csvs(root)
    if task_csvs:
        print("\n--- Detected per-task class-index CSVs ---")
        for task, files in task_csvs.items():
            print(f"  {task}: {[str(f.relative_to(root)) for f in files]}")

    # 4b. Detected steubk/wikiart-style layout (folder-per-style + a
    # readable classes.csv-style metadata file for artist/genre)
    readable = None if task_csvs else _find_readable_classes_csv(root)
    is_folder_layout = _looks_like_style_folder_layout(root)
    if readable and is_folder_layout:
        csv_path, col_path, col_artist, col_genre = readable
        print(f"\n--- Detected steubk/wikiart layout ---")
        print(f"  Style: from folder names")
        print(f"  Artist/genre: from {csv_path.relative_to(root)} "
              f"(columns: path={col_path}, artist={col_artist}, genre={col_genre})")

    # 5. Image files and folder depth
    images = []
    for ext in IMAGE_EXTS:
        images.extend(root.rglob(f"*{ext}"))
        if len(images) > 2000:
            break
    print(f"\n--- Image files found (sampled, capped scan): {len(images)}+ ---")
    if images:
        depths = {len(p.relative_to(root).parts) for p in images[:200]}
        print(f"  Path depth(s) relative to root seen in sample: {sorted(depths)}")
        print(f"  Sample filenames:")
        for p in images[:sample_n]:
            print(f"    {p.relative_to(root)}")

    # 6. Guess layout
    print("\n--- Guess ---")
    if task_csvs:
        print("  Looks like a per-task class-index CSV layout. data_prep.py")
        print("  will use these automatically. No date/year field exists in")
        print("  this layout, so the 'era' task will be empty — use 'genre'")
        print("  as the third task instead (or merge in dates yourself).")
    elif readable and is_folder_layout:
        print("  Looks like the steubk/wikiart layout: folder-per-style +")
        print("  a classes.csv-style metadata file. data_prep.py will use")
        print("  folder names for style and that CSV for artist/genre")
        print("  automatically. No date/year field exists in this dataset,")
        print("  so use 'genre' as the third task instead of 'era'.")
    elif csvs:
        print("  Looks CSV-driven (single metadata CSV). Check column mapping")
        print("  in COLUMN_ALIASES against the columns printed above.")
    elif images and len({len(p.relative_to(root).parts) for p in images[:200]}) == 1 and any(d.is_dir() for d in entries):
        print("  Looks like folder-per-class (e.g. root/<Style>/<file>.jpg).")
        print("  Artist attribution / era will rely on filename parsing,")
        print("  which is fragile — verify against the sample filenames above.")
    else:
        print("  Could not confidently guess. Inspect manually and adjust")
        print("  build_metadata() accordingly.")


def _find_column(df: pd.DataFrame, aliases: list) -> str | None:
    lower_cols = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_cols:
            return lower_cols[alias]
    return None


def _parse_filename_fallback(filename: str) -> dict:
    """
    Best-effort parse for the common WikiArt filename convention:
    'artist-name_title-of-the-work.jpg' (no year, typically).
    Returns {'artist': str|None}. This is a guess — verify against your
    actual filenames from --inspect before trusting it.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 1:
        artist_raw = parts[0]
        artist = artist_raw.replace("-", " ").strip().title()
        return {"artist": artist if artist else None}
    return {"artist": None}


# ---------------------------------------------------------------------------
# Path (a): steubk/wikiart-style per-task class-index CSVs
# ---------------------------------------------------------------------------

def _find_task_csvs(root: Path) -> dict:
    """
    Look for CSVs whose filename mentions one of TASK_KEYWORDS together with
    'train' or 'val'/'test', e.g. style_train.csv, artist_val.csv,
    genre_train.csv. Returns {task: [Path, ...]}, only for tasks that have
    at least one match. Filenames are matched case-insensitively and are
    fairly permissive (e.g. 'wikiart_style_train.csv' also matches).
    """
    found = {}
    for csv_path in root.rglob("*.csv"):
        name = csv_path.name.lower()
        for task in TASK_KEYWORDS:
            if task in name and re.search(r"train|val|test", name):
                found.setdefault(task, []).append(csv_path)
    return found


def _find_class_names(root: Path, task: str) -> dict | None:
    """
    Look for a class-index -> class-name lookup for `task`, e.g.
    style_class.txt / style_classes.txt / classes_style.csv. Accepts
    whitespace- or comma-separated "idx name" per line, with or without a
    header. Returns {idx: name} or None if nothing found/parseable.
    """
    candidates = []
    for ext_glob in ("*.txt", "*.csv"):
        for p in root.rglob(ext_glob):
            name = p.name.lower()
            if task in name and "class" in name:
                candidates.append(p)

    for p in candidates:
        try:
            mapping = {}
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = re.split(r"[,\t]|\s+", line, maxsplit=1)
                    if len(parts) != 2:
                        continue
                    idx_str, label = parts
                    if not idx_str.strip("-").isdigit():
                        continue
                    mapping[int(idx_str)] = label.strip().strip('"')
            if mapping:
                return mapping
        except Exception:
            continue
    return None


def _resolve_image_path(root: Path, rel_path: str, basename_index: dict) -> str | None:
    """
    Try a few strategies to turn a path recorded in one of these CSVs into
    an actual file on disk, since the recorded prefix (e.g. 'train/',
    'wikiart/') doesn't always match the extracted folder layout:
      1. root / rel_path as given
      2. root / rel_path with the first path component stripped
      3. basename lookup against a precomputed index of every image on disk
    Returns None if nothing resolves (row will be dropped later).
    """
    rel_path = rel_path.strip().lstrip("/\\").replace("\\", "/")

    direct = root / rel_path
    if direct.exists():
        return str(direct)

    parts = rel_path.split("/", 1)
    if len(parts) == 2:
        stripped = root / parts[1]
        if stripped.exists():
            return str(stripped)

    basename = os.path.basename(rel_path)
    hit = basename_index.get(basename)
    if hit:
        return str(hit)

    return None


def _build_basename_index(root: Path) -> dict:
    print("  Indexing image files on disk for path resolution (one-time scan)...")
    index = {}
    for ext in IMAGE_EXTS:
        for p in root.rglob(f"*{ext}"):
            index.setdefault(p.name, p)
    print(f"  Indexed {len(index)} image files.")
    return index


def build_from_task_csvs(root: str, task_csvs: dict) -> pd.DataFrame:
    """
    Build a unified metadata frame from steubk/wikiart-style per-task
    class-index CSVs. Each task's train+val CSVs are concatenated (we do
    our own stratified split downstream in dataset.py, so the original
    train/val split doesn't need to be preserved).
    """
    root = Path(root)
    basename_index = _build_basename_index(root)

    per_task_frames = {}
    for task, files in task_csvs.items():
        class_names = _find_class_names(root, task)
        if class_names is None:
            print(f"  No class-name lookup found for '{task}'; using raw "
                  f"integer class indices as labels (rename these later if "
                  f"you find the lookup file).")

        rows = []
        for f in files:
            try:
                df = pd.read_csv(f, header=None, names=["rel_path", "class_idx"])
            except Exception as e:
                print(f"  Skipping {f} (couldn't parse as path,class_idx CSV: {e})")
                continue
            for rel_path, class_idx in zip(df["rel_path"], df["class_idx"]):
                try:
                    idx = int(class_idx)
                except (ValueError, TypeError):
                    continue
                label = class_names.get(idx, str(idx)) if class_names else str(idx)
                resolved = _resolve_image_path(root, str(rel_path), basename_index)
                if resolved is None:
                    continue
                rows.append({"path": resolved, task: label})

        task_df = pd.DataFrame(rows)
        n_unresolved = sum(len(pd.read_csv(f, header=None)) for f in files) - len(task_df)
        print(f"  [{task}] {len(task_df)} rows resolved to files on disk "
              f"({max(n_unresolved, 0)} rows dropped: path didn't resolve).")
        per_task_frames[task] = task_df.drop_duplicates(subset=["path"])

    # Outer-join all task frames on path, so an image gets whichever labels
    # exist for it (e.g. it may appear in style_* and artist_* but not
    # genre_*, if a task's CSVs don't cover every image).
    merged = None
    for task, task_df in per_task_frames.items():
        merged = task_df if merged is None else merged.merge(task_df, on="path", how="outer")

    for col in ("artist", "style", "genre"):
        if col not in merged.columns:
            merged[col] = pd.NA
    merged["year"] = pd.NA  # steubk/wikiart has no date field at all

    return merged[["path", "artist", "style", "genre", "year"]]


# ---------------------------------------------------------------------------
# Path (b): confirmed steubk/wikiart layout — folder-per-style for the style
# label, joined with a "classes.csv"-style metadata file for artist/genre.
# ---------------------------------------------------------------------------

def _looks_like_style_folder_layout(root: Path) -> bool:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return False
    return any(
        any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir() if f.is_file())
        for d in dirs[:3]
    )


def _find_readable_classes_csv(root: Path):
    """
    Look for a CSV with a genuinely human-readable artist column (plain
    names) paired with a path/filename column, as opposed to a CSV like
    wclasses.csv whose 'artist' column is actually an integer class code
    with no shipped name lookup. Returns (csv_path, col_path, col_artist,
    col_genre) for the first match, or None.
    """
    for csv_path in sorted(root.rglob("*.csv")):
        try:
            sample = pd.read_csv(csv_path, nrows=20)
        except Exception:
            continue
        col_path = _find_column(sample, COLUMN_ALIASES["path"])
        col_artist = _find_column(sample, COLUMN_ALIASES["artist"])
        col_genre = _find_column(sample, COLUMN_ALIASES["genre"])
        if col_path is None or col_artist is None:
            continue
        artist_vals = sample[col_artist].dropna().astype(str).head(10)
        if len(artist_vals) and artist_vals.str.fullmatch(r"-?\d+").all():
            continue  # looks like integer codes, e.g. wclasses.csv
        return csv_path, col_path, col_artist, col_genre
    return None


def _parse_genre_field(raw):
    """
    steubk/wikiart's classes.csv stores genre as a stringified Python list
    of WikiArt genre tags, e.g. "['Abstract Expressionism']" or
    "['Portrait', 'Still Life']" (an artwork can have several tags; this
    pipeline is single-label, so we take the first). Falls through
    gracefully for mirrors where genre is already a plain string.

    In practice, for this dataset classes.csv's genre is almost always
    just a copy of the style name (not a real distinct genre tag) — see
    _apply_wclasses_genre_override, which prefers a real genre-code source
    when one is available and only falls back to this text for the small
    minority of images that have a genuinely distinct tag here.
    """
    if pd.isna(raw):
        return pd.NA
    raw = str(raw).strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            tags = ast.literal_eval(raw)
            if isinstance(tags, (list, tuple)) and tags:
                return str(tags[0]).strip()
            return pd.NA
        except (ValueError, SyntaxError):
            cleaned = re.sub(r"[\[\]'\"]", "", raw).split(",")[0].strip()
            return cleaned or pd.NA
    return raw or pd.NA


# The steubk/wikiart download's classes.csv genre field turns out to be
# almost entirely a copy of the style name (97.5%+ of rows), not a real
# distinct genre tag — confirmed by cross-referencing wclasses.csv's
# integer genre codes against folder-derived style. wclasses.csv's genre
# codes, on the other hand, line up with the well-documented 10-class
# ArtGAN/WikiArt genre taxonomy (Abstract Painting, Cityscape, Genre
# Painting, Illustration, Landscape, Nude Painting, Portrait, Religious
# Painting, Sketch and Study, Still Life) plus an 11th "Unknown Genre"
# bucket, confirmed both by the published class list and by strong
# internal consistency (e.g. the code dominated by Cubism/Synthetic Cubism
# images maps to Still Life — a well-known real association — and the
# code dominated by Art Nouveau maps to Illustration, likewise real).
CANONICAL_GENRE_NAMES = [
    "Abstract Painting", "Cityscape", "Genre Painting", "Illustration",
    "Landscape", "Nude Painting", "Portrait", "Religious Painting",
    "Sketch and Study", "Still Life", "Unknown Genre",
]


def _find_numeric_genre_csv(root: Path, exclude: Path):
    """
    Look for a sibling CSV (e.g. wclasses.csv) with a numeric genre-code
    column and a path column, to use as the *real* genre source instead of
    classes.csv's mostly-duplicate text. Returns (csv_path, col_path,
    col_genre) or None.
    """
    for csv_path in sorted(root.rglob("*.csv")):
        if csv_path == exclude:
            continue
        try:
            sample = pd.read_csv(csv_path, nrows=20)
        except Exception:
            continue
        col_path = _find_column(sample, COLUMN_ALIASES["path"])
        col_genre = _find_column(sample, COLUMN_ALIASES["genre"])
        if col_path is None or col_genre is None:
            continue
        vals = sample[col_genre].dropna().astype(str).head(10)
        if len(vals) and vals.str.fullmatch(r"-?\d+").all():
            return csv_path, col_path, col_genre
    return None


def apply_wclasses_genre_override(df: pd.DataFrame, root: str, exclude_csv: Path) -> pd.DataFrame:
    """
    Overlay a real genre label onto `df` (which already has a
    classes.csv-derived `genre` column, mostly duplicating `style`):
      1. If a sibling numeric genre-code CSV exists (e.g. wclasses.csv) and
         it has exactly len(CANONICAL_GENRE_NAMES) unique codes, map codes
         -> canonical names by sorted position and use that wherever a
         code is available for the image.
      2. Otherwise, for images without a code, keep classes.csv's genre
         text ONLY if it's not just a copy of that image's style (i.e. a
         genuinely distinct tag) — else the label is left blank, since a
         style-duplicate isn't real genre information.
    """
    found = _find_numeric_genre_csv(Path(root), exclude_csv)
    df = df.copy()
    is_dupe = df["genre"].astype(str).str.lower() == df["style"].astype(str).str.lower()
    df.loc[is_dupe, "genre"] = pd.NA  # drop the uninformative style-copies

    if found is None:
        print("  No numeric genre-code CSV found alongside the metadata CSV; "
              "genre will only be populated for images with a genuinely "
              "distinct tag in the metadata CSV (likely a small minority).")
        return df

    code_csv, code_col_path, code_col_genre = found
    codes_df = pd.read_csv(code_csv)
    codes_df[code_col_path] = (
        codes_df[code_col_path].astype(str).str.strip().str.lstrip("/\\").str.replace("\\", "/", regex=False)
    )
    unique_codes = sorted(codes_df[code_col_genre].dropna().unique())

    if len(unique_codes) != len(CANONICAL_GENRE_NAMES):
        print(f"  Found {code_csv.name} with a numeric genre column, but it has "
              f"{len(unique_codes)} unique codes (expected {len(CANONICAL_GENRE_NAMES)} "
              f"to confidently map to real genre names) — leaving genre as-is "
              f"rather than guessing. Inspect {code_csv.name} manually if you "
              f"want to map these codes yourself.")
        return df

    code_to_name = dict(zip(unique_codes, CANONICAL_GENRE_NAMES))
    print(f"  Using {code_csv.name}'s numeric genre codes as the real genre "
          f"label (mapped to the standard 10-genre + Unknown taxonomy): "
          f"{code_to_name}")

    rel_to_path = {}
    root = Path(root)
    for p in df["path"]:
        rel_to_path[str(Path(p).relative_to(root)).replace("\\", "/")] = p

    code_lookup = dict(zip(codes_df[code_col_path], codes_df[code_col_genre]))
    path_to_genre_name = {}
    for rel, full_path in rel_to_path.items():
        code = code_lookup.get(rel)
        if code in code_to_name:
            path_to_genre_name[full_path] = code_to_name[code]

    n_overridden = sum(1 for p in df["path"] if p in path_to_genre_name)
    df["genre"] = df["path"].map(path_to_genre_name).combine_first(df["genre"])
    print(f"  Applied real genre codes to {n_overridden} / {len(df)} images "
          f"(remaining images keep any genuinely distinct classes.csv tag, "
          f"or are left blank if neither source has one).")
    return df


def build_from_steubk_wikiart(root: str, csv_path: Path, col_path: str,
                               col_artist: str, col_genre: str | None) -> pd.DataFrame:
    root = Path(root)
    meta = pd.read_csv(csv_path)
    meta[col_path] = meta[col_path].astype(str).str.strip().str.lstrip("/\\").str.replace("\\", "/", regex=False)
    if col_genre:
        meta["_genre_parsed"] = meta[col_genre].apply(_parse_genre_field)
    else:
        meta["_genre_parsed"] = pd.NA

    # Build lookup dicts: exact relative path first, basename as a fallback
    # for filename edge cases (dedupe basenames so we don't guess wrong).
    by_relpath = meta.set_index(col_path)[[col_artist, "_genre_parsed"]].to_dict("index")
    basename_counts = meta[col_path].apply(lambda p: os.path.basename(p)).value_counts()
    unique_basenames = set(basename_counts[basename_counts == 1].index)
    by_basename = {}
    for p, row in by_relpath.items():
        b = os.path.basename(p)
        if b in unique_basenames:
            by_basename[b] = row

    rows = []
    n_no_lookup = 0
    for style_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        style = style_dir.name.replace("_", " ")
        for img_path in style_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = str(img_path.relative_to(root)).replace("\\", "/")
            match = by_relpath.get(rel) or by_basename.get(img_path.name)
            if match is not None:
                artist, genre = match[col_artist], match["_genre_parsed"]
            else:
                n_no_lookup += 1
                parsed = _parse_filename_fallback(img_path.name)
                artist, genre = parsed["artist"], pd.NA
            rows.append({
                "path": str(img_path),
                "artist": artist,
                "style": style,
                "genre": genre,
                "year": pd.NA,  # steubk/wikiart has no date field at all
            })

    if n_no_lookup:
        print(f"  {n_no_lookup} images had no {csv_path.name} entry; artist "
              f"parsed from filename as a fallback, genre left blank for them.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Path (c): single unified CSV with recognizable columns
# ---------------------------------------------------------------------------

def _score_csv_for_metadata(csv_path: Path) -> tuple:
    """
    Score a candidate single CSV by how many of (path, artist, style, genre)
    columns it has, so when multiple metadata CSVs exist at the root (e.g.
    steubk/wikiart's classes.csv + wclasses.csv) we pick the one that
    actually covers our three tasks instead of an arbitrary filesystem-order
    first match. Returns (n_columns_covered, has_style) for sorting;
    has_style is a tiebreaker since style is required and some sibling
    CSVs in this dataset omit it entirely.
    """
    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return (-1, False)
    found = {
        key: _find_column(df, aliases) is not None
        for key, aliases in COLUMN_ALIASES.items()
    }
    return (sum(found.values()), found["style"])


def build_from_csv(csv_path: str, root: str, basename_index: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    col_artist = _find_column(df, COLUMN_ALIASES["artist"])
    col_style = _find_column(df, COLUMN_ALIASES["style"])
    col_genre = _find_column(df, COLUMN_ALIASES["genre"])
    col_year = _find_column(df, COLUMN_ALIASES["year"])
    col_path = _find_column(df, COLUMN_ALIASES["path"])

    missing = [name for name, col in
               [("artist", col_artist), ("style", col_style), ("path", col_path)]
               if col is None]
    if missing:
        raise ValueError(
            f"Could not find columns for: {missing}. "
            f"Actual columns are: {list(df.columns)}. "
            f"Edit COLUMN_ALIASES in src/data_prep.py to map them, or if "
            f"another CSV in this dataset has the missing column(s), rerun "
            f"pointing --root at a layout where that CSV is picked (see "
            f"_score_csv_for_metadata in this file for the selection logic)."
        )

    root = Path(root)
    if basename_index is None:
        basename_index = _build_basename_index(root)

    def _resolve(p: str) -> str | None:
        p = str(p)
        if os.path.isabs(p) and os.path.exists(p):
            return p
        return _resolve_image_path(root, p, basename_index)

    resolved_paths = df[col_path].apply(_resolve)
    n_unresolved = resolved_paths.isna().sum()
    if n_unresolved:
        print(f"  {n_unresolved} / {len(df)} rows had a path that couldn't be "
              f"matched to a file on disk and will be dropped.")

    out = pd.DataFrame()
    out["path"] = resolved_paths
    out["artist"] = df[col_artist]
    out["style"] = df[col_style]
    out["genre"] = df[col_genre] if col_genre else pd.NA
    out["year"] = pd.to_numeric(df[col_year], errors="coerce") if col_year else pd.NA

    return out.dropna(subset=["path"])


# ---------------------------------------------------------------------------
# Path (c): folder-per-style + filename parsing
# ---------------------------------------------------------------------------

def build_from_folders(root: str) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for style_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        style = style_dir.name.replace("_", " ")
        for img_path in style_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            parsed = _parse_filename_fallback(img_path.name)
            rows.append({
                "path": str(img_path),
                "artist": parsed["artist"],
                "style": style,
                "genre": pd.NA,  # not recoverable from this layout
                "year": pd.NA,   # not recoverable from this layout
            })
    return pd.DataFrame(rows)


def add_era_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    year = pd.to_numeric(df["year"], errors="coerce").clip(
        config.ERA_MIN_YEAR, config.ERA_MAX_YEAR
    )
    bin_start = (year // config.ERA_BIN_SIZE) * config.ERA_BIN_SIZE
    df["era_label"] = bin_start.apply(
        lambda b: f"{int(b)}-{int(b) + config.ERA_BIN_SIZE - 1}" if pd.notna(b) else pd.NA
    )
    return df


def filter_rare_classes(df: pd.DataFrame, col: str, min_count: int) -> pd.DataFrame:
    """
    Drops rows whose `col` label is a rare class (fewer than min_count
    occurrences). Rows where `col` is simply NaN (this task's label was
    never available for that image, e.g. genre in a folder-per-style
    layout) are left alone here — they're excluded per-task in
    dataset.py's load_splits(), not here.
    """
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    is_rare = df[col].isin(rare)
    if is_rare.any():
        print(f"  Dropping {int(is_rare.sum())} rows: '{col}' classes with < {min_count} samples.")
    return df[~is_rare].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to WikiArt dataset root")
    parser.add_argument("--inspect", action="store_true", help="Just print structure and exit")
    parser.add_argument("--out", default="metadata.csv", help="Output metadata CSV path")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.root)
        return

    root_path = Path(args.root)
    task_csvs = _find_task_csvs(root_path)
    is_folder_layout = _looks_like_style_folder_layout(root_path)
    readable_csv = None if task_csvs else _find_readable_classes_csv(root_path)
    single_csvs = [] if (task_csvs or readable_csv) else list(root_path.rglob("*.csv"))

    if task_csvs:
        print(f"Detected per-task class-index CSVs for: {list(task_csvs)}")
        df = build_from_task_csvs(args.root, task_csvs)
    elif readable_csv and is_folder_layout:
        csv_path, col_path, col_artist, col_genre = readable_csv
        print(f"Detected folder-per-style layout + a readable metadata CSV "
              f"({csv_path.name}). Using folder names for style and "
              f"{csv_path.name} for artist/genre (this is the steubk/wikiart "
              f"layout — any wclasses.csv-style integer-coded sibling CSV is "
              f"ignored since it has no shipped name lookup).")
        df = build_from_steubk_wikiart(args.root, csv_path, col_path, col_artist, col_genre)
        df = apply_wclasses_genre_override(df, args.root, exclude_csv=csv_path)
    elif single_csvs:
        if len(single_csvs) > 1:
            scored = sorted(single_csvs, key=_score_csv_for_metadata, reverse=True)
            print(f"Found {len(single_csvs)} CSVs: {[c.name for c in scored]}")
            print(f"Using {scored[0].name} (best column coverage for path/artist/style/genre).")
            chosen = scored[0]
        else:
            chosen = single_csvs[0]
            print(f"Found CSV, using {chosen.name} to build metadata.")
        df = build_from_csv(str(chosen), args.root)
    else:
        print("No CSV found, falling back to folder-per-style + filename parsing.")
        df = build_from_folders(args.root)

    n_before = len(df)
    # Only path is truly required; a row missing one task's label just
    # can't be used for that task (handled per-task in dataset.py), but we
    # still want at least SOME usable label to keep the row.
    df = df.dropna(subset=["path"])
    df = df[df[["artist", "style", "genre"]].notna().any(axis=1)]
    print(f"Rows with a usable path and at least one label: {len(df)} / {n_before}")

    df = add_era_label(df)
    n_with_year = df["era_label"].notna().sum()
    print(f"Rows with usable year/era: {n_with_year} / {len(df)}")
    if n_with_year == 0:
        print("  WARNING: no dates found anywhere. Era task will not be trainable")
        print("  until you merge in a metadata source that has dates. If you're")
        print("  using steubk/wikiart, use --task genre instead of --task era.")

    df = filter_rare_classes(df, "style", config.MIN_SAMPLES_PER_STYLE)
    df = filter_rare_classes(df, "artist", config.MIN_SAMPLES_PER_ARTIST)
    df = filter_rare_classes(df, "genre", config.MIN_SAMPLES_PER_GENRE)

    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")
    print(f"  Unique styles: {df['style'].nunique()}")
    print(f"  Unique artists: {df['artist'].nunique()}")
    print(f"  Unique genres: {df['genre'].nunique()}")
    print(f"  Unique eras: {df['era_label'].nunique()}")


if __name__ == "__main__":
    main()