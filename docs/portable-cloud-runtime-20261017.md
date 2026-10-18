# Locked cloud runtime and external COOL SDK

The primary workload must run on Linux ARM64 AWS Graviton using OpenCV5 + COOL.
The evaluated configuration used an m8g.4xlarge, CLASSIC inference and eight OpenCV
threads. Python dependencies are pinned by uv.lock. SDK binaries, model weights
and cloud credentials are excluded from this repository.

## Separate cloud environment

The entitled COOL SDK must already be installed separately. Installing it from a
blank host and compatibility with a different native SDK build remain unvalidated.
Do not substitute a standard OpenCV wheel for evidence of COOL execution.

```sh
UV_PROJECT_ENVIRONMENT=.venv-cloud uv sync --frozen --no-default-groups
PYTHONPATH=/opt/cool/python_3.11/site-packages LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib .venv-cloud/bin/python -c 'from care_vision.common.cool import verify_runtime; print(verify_runtime()["verified"])'
```

These are generic SDK locations, not account-specific deployment details. Source
runtime validation checks architecture and native provenance. SDK entitlement,
exact library versions and approved models/configuration must be verified on the
chosen host before using the inference command.

## Recorded comparison runner

`tools/run_portable_classic_replays.py` accepts explicit host, SSH identity, remote
roots, Python/uv executables, external assets and a frozen profile. It creates a
fresh isolated locked environment, runs bounded recorded cases, records provenance,
audit logs and owned-worker cleanup. It leaves the instance running. The required
uv version is declared in `src/care_vision/aws_demo.py` (0.11.32 in this snapshot).

The experimental wrapper requires `CARE_DIAGNOSTIC_BYTE_BUDGET=114688`; that selects
WebP80 and regular full-frame sampling. The local proxy is an evaluation tool, not
a new AWS service. A prepare-only command is provided in the root README.

For an authorized run, supply all documented CLI arguments and omit prepare-only.
Use an existing trusted SSH alias and key outside the checkout. Never disable
StrictHostKeyChecking. Do not commit the private asset/profile bundle, logs or
state outputs. Run the matching original audit against the exact source used by
the worker. Provisioning a public endpoint, uploading recordings and contacting
caregivers are not part of this runner.
