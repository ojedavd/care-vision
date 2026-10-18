# Security and private material

This is a code-only repository. Personal test recordings, original inputs,
photos, audio, logs, state snapshots, camera credentials, SSH keys, account IDs
and private deployment addresses belong in a separate local workspace.

## Before a commit or push

1. Install Gitleaks and enable the versioned hooks with
   `git config --local core.hooksPath .githooks`.
2. Review `git diff --cached --stat` and run the staged hygiene check.
3. Run `gitleaks git --pre-commit --staged --ignore-gitleaks-allow --redact=100`.
4. Before publishing history, run `python3 tools/check_repository_hygiene.py --history`
   and `gitleaks git --log-opts=--all --ignore-gitleaks-allow --redact=100`.

Do not bypass hooks to add a recording or credential. Ignore patterns do not
remove an already tracked file. Keep credentials in external owner-controlled
storage; provide connection destinations and key paths explicitly to tools.
The historical utilities that used author-specific connection defaults now
require `CARE_VISION_SSH_HOST` / `CARE_VISION_SSH_IDENTITY` or explicit CLI options.

Hygiene checks are a defense in depth and do not certify that arbitrary prose or
application data are anonymous. Review every addition. Hook configuration is local
to a clone and must be enabled again in a new clone. CI repeats the source-only
check and synthetic tests, with read-only repository permissions and no AWS secrets.

## New recordings

Use a private absolute directory outside this checkout for all capture outputs.
Do not attach personal recordings to issues, pull requests, releases or Actions
artifacts. Presentation blur is post-processing and does not anonymize originals.
Keep scene calibration and reserved evaluation separate; declare the profile and
annotations before examining reserved predictions. No real fall enactment is
needed for the proposed everyday-activity negative clips.

## Reporting a problem

Contact the repository owner through the existing private collaboration channel.
Do not post credentials, personal visual evidence or state dumps in an issue.
If an actual credential has been exposed, revoke/rotate it before sharing any
sanitized diagnostic report; removing the current file does not remove history.
