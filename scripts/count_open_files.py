import os
import psutil


def count_total_open_files() -> int:
    total_files = 0
    for proc in psutil.process_iter(["pid"]):
        pid = proc.info["pid"]
        fd_dir = f"/proc/{pid}/fd"
        try:
            files = os.listdir(fd_dir)
            total_files += len(files)
        except (FileNotFoundError, PermissionError):
            continue  # Process may have exited or access denied
    return total_files


if __name__ == "__main__":
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    total = count_total_open_files()
    logger.info("Total open files across all processes: %s", total)
