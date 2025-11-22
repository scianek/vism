import warnings

# Suppress warnings from DINOv2 (https://github.com/facebookresearch/dinov2/issues/513)
warnings.filterwarnings("ignore", message="xFormers is (available|not available)")
