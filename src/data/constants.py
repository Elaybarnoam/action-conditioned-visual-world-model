"""Source and format constants for the first BAIR working subset."""

ARCHIVE_URL = "https://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar"
TFDS_READER_URL = (
    "https://raw.githubusercontent.com/tensorflow/datasets/master/"
    "tensorflow_datasets/datasets/bair_robot_pushing_small/"
    "bair_robot_pushing_small_dataset_builder.py"
)
ORIGINAL_READER_URL = (
    "https://raw.githubusercontent.com/febert/visual_mpc/master/"
    "python_visual_mpc/video_prediction/read_tf_record_sawyer12.py"
)
CONTROLLER_URL = (
    "https://raw.githubusercontent.com/febert/visual_mpc/master/"
    "python_visual_mpc/sawyer/visual_mpc_rospkg/src/visual_mpc_client.py"
)
FRAMES = 30
IMAGE_SHAPE = (64, 64, 3)
ACTION_SIZE = 4
POSITION_SIZE = 3
CAMERAS = ("image_main", "image_aux1")
SCHEMA_VERSION = 1
VALIDATION_FRACTION = 0.2
MAX_RECORD_BYTES = 2_000_000
