from pathlib import Path

from comet_ml import comet
from ray_utilities.comet import CometArchiveTracker
from ray_utilities.constants import COMET_OFFLINE_DIRECTORY
import dotenv

dotenv.load_dotenv("~/.comet_api_key.env")


files = [
    "../outputs/.cometml-runs/uploaded/37a9520a726a41a89dab41689ce1678b.zip",
]

for p in list(map(Path, files)):
    if not p.exists():
        print(f"File {p} does not exist")

paths = [p for f in files if (p := Path(f)).exists()]

# Note can be further simplified by extending comet_upload_offline_experiments
tracker = CometArchiveTracker(track=paths, path=Path(COMET_OFFLINE_DIRECTORY), auto=True)
tracker.upload_and_move()
