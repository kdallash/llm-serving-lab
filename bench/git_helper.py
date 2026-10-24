import subprocess
def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"