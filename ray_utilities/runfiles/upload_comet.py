from pathlib import Path

from ray_utilities.comet import CometArchiveTracker
from ray_utilities.constants import COMET_OFFLINE_DIRECTORY
import dotenv
from ray_utilities.nice_logger import nice_logger

dotenv.load_dotenv("~/.comet_api_key.env")


logger = nice_logger(__name__)

files = [
    "../outputs/.cometml-runs/uploaded/37a9520a726a41a89dab41689ce1678b.zip",
]

paths = []
for f in files:
    p = Path(f)
    if not p.exists():
        logger.warning("File %s does not exist", p)
    else:
        paths.append(p)

# Note can be further simplified by extending comet_upload_offline_experiments
tracker = CometArchiveTracker(track=paths, path=Path(COMET_OFFLINE_DIRECTORY), auto=True)
tracker.upload_and_move()
