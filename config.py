"""
Central config. Tweak these once you see your real dataset via
`python -m src.data_prep --inspect`.

Pinned to: https://www.kaggle.com/datasets/steubk/wikiart
This is the ~81k-image WikiArt archive (same underlying source as the
huggan/wikiart / ArtGAN "new WikiArt dataset"). It ships as:
  - images grouped in style folders, and
  - a set of per-task class-index CSVs (style/artist/genre train+val splits
    mapping image path -> integer class index) plus class-name lookup files.
`data_prep.py` auto-detects the CSV layout; run --inspect first if your local copy looks different.
"""

# --- Image handling ---
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Splitting ---
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

# --- Artist attribution ---
# Artists with fewer images than this are dropped entirely (unlearnable
# classes, and they break stratified splitting since sklearn needs >= 2
# samples per class per split).
MIN_SAMPLES_PER_ARTIST = 10

# --- Style ---
MIN_SAMPLES_PER_STYLE = 10

# --- Genre ---
# The steubk/wikiart Kaggle dataset ships genre labels (~11 classes,
# including "Unknown Genre"); this is the third task for that dataset.
MIN_SAMPLES_PER_GENRE = 10

# --- Era binning ---
# Used only if --era-mode binned (the default), and only if your metadata
# actually has a year/date column. The steubk/wikiart dataset does NOT
# include dates, so era_label will end up all-empty and the "era" task
# will refuse to train (data_prep.py will warn you at build time) unless
# you merge in a metadata source that has dates yourself. Bin edges below
# are in years.
ERA_BIN_SIZE = 50          # 50-year buckets, e.g. 1800-1849, 1850-1899, ...
ERA_MIN_YEAR = 1300        # anything earlier gets clipped into the first bin
ERA_MAX_YEAR = 2000        # anything later gets clipped into the last bin

# --- Training defaults (overridable via CLI flags) ---
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 10
BACKBONE = "resnet50"      # see src/model.py for supported options

# --- Paths ---
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"