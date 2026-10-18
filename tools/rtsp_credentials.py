"""Store RTSP camera credentials privately and run a command with them."""

import argparse
import getpass
import json
import os
import subprocess
from pathlib import Path

CREDENTIAL_FILE = Path.home() / ".config/care-vision/rtsp-credentials.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("store")
    run = subparsers.add_parser("run")
    run.add_argument("command", nargs=argparse.REMAINDER)
    subparsers.add_parser("delete")
    args = parser.parse_args()

    if args.action == "store":
        username = input("Camera account username: ").strip()
        password = getpass.getpass("Camera account password: ")
        if not username or not password:
            parser.error("username and password must be nonempty")
        CREDENTIAL_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(CREDENTIAL_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"username": username, "password": password}, stream)
                stream.write("\n")
        except BaseException:
            CREDENTIAL_FILE.unlink(missing_ok=True)
            raise
        print("Credentials saved locally with owner-only access.")
    elif args.action == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("run requires a command after --")
        data = json.loads(CREDENTIAL_FILE.read_text(encoding="utf-8"))
        env = os.environ.copy()
        env["CARE_VISION_RTSP_USERNAME"] = data["username"]
        env["CARE_VISION_RTSP_PASSWORD"] = data["password"]
        raise SystemExit(subprocess.call(command, env=env))
    else:
        CREDENTIAL_FILE.unlink(missing_ok=True)
        print("Local RTSP credential file deleted.")


if __name__ == "__main__":
    main()
