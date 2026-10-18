# Code-only source review

The GitHub tree is the reviewed publication copy. It intentionally excludes videos,
images/audio, input/model bundles, logs, snapshots, SDK binaries and credentials.
The primary source matches the selected experimental candidate; publication changes
and provenance are described in the root README and docs/source-provenance.json.

Use the build, focused data-free test and cloud-environment commands in README.md.
Historical tools/tests may depend on private artifacts; the full suite is not a
supported data-free acceptance command. A source-only clone cannot reproduce real
recordings without independently authorized external inputs and their frozen profile.

Before creating a source export, run the repository hygiene and Gitleaks checks.
Use Git's reviewed file set rather than archiving the original experiment directory:

```sh
python3 tools/check_repository_hygiene.py --history
gitleaks git --log-opts=--all --ignore-gitleaks-allow --redact=100
# Choose an absolute output path outside the checkout:
git archive --format=zip --output=/PRIVATE/care-vision-source.zip HEAD
```

This command is local and grants no judge access by itself. Revalidate the exported
source and instructions before sending it. The older build_submission_source.py
helper is retained for historical reproducibility; its marker check is not a complete
secret/privacy audit. No open-source license or redistribution permission is selected
automatically by creating a private repository.
