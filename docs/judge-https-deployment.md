# HTTPS entry for the private judge portal — deployed and externally verified

## Experimental seated closure profile

The portal can explicitly enable the evaluated seated-pattern controller using
`experimental-seated-closure-seconds` in its private cloud-options JSON. This
requires `experimental-s-only-5fps: true` and a `pose-shadow-config` file for the
reviewed camera/chair scene. The pose model and scene configuration remain outside
Git. Use a finite positive numeric duration; five seconds is an uncalibrated
development value. This does not change the default cloud policy.

The worker reports the active pose/closure profile after startup, and the web
projection shows an experimental seated-pattern resolution in its timeline.
No configuration paths, tracks, pose keypoints or images are exposed. Pose ROI
cycles and alerts are separate options; neither is needed for this closure path.
Starting a cloud session still requires a separately connected local Edge.

Scene regions need review after camera/chair movement. This profile has only
development evidence for a small set of activities; it does not verify physical
seat support, health or general fall accuracy. Keep previous per-session logs and
protected episode snapshots separate when changing profiles.

The deployed entry is CloudFront with an AWS-managed hostname and an EC2 VPC
origin. It forwards only the authenticated, image-free judge HTTP application.
The vision worker, Edge command channel and COOL workload retain their existing
Graviton/SSH boundary. No input recordings, model files, SDK or private evidence
are part of the CloudFormation plan.

This adds one managed HTTPS entry because no owned DNS name has been supplied.
CloudFront traffic has its own pricing; this plan does not assert that the compute
grant covers it or that deployment is free. Local preparation created no AWS resources. A later authenticated deployment
created a disabled stack and a separate supervised origin, then enabled the
restricted external path after review. External acceptance has passed within the
image-free review scope below. The existing private SSH portal remains available.

## Generate and validate without AWS

```sh
uv sync --frozen --group https-check
uv run --frozen python tools/prepare_judge_https.py --output /PRIVATE/new-https-plan
uv run --frozen --group https-check cfn-lint /PRIVATE/new-https-plan/cloudfront-vpc-origin.json
```

The optional checker and its transitive dependencies are locked by `uv.lock`.
The generator refuses a destination inside the code repository or an existing
output directory. It makes no network/AWS calls. The template has no account,
instance or VPC defaults. Supply identifiers only in a private deployment session.

## Resources and protective defaults

- CloudFront distribution with the default AWS-managed HTTPS hostname/certificate.
  It starts disabled and cannot be enabled by the template unless the managed
  CloudFront origin security group is also explicitly supplied.
- CloudFront VPC origin referencing the existing instance ARN and private DNS.
  Origin traffic uses the private VPC path, not a public HTTP origin.
- A dedicated portal security group, initially with no ingress. Its only optional
  ingress rule is TCP8780 from the verified AWS-managed CloudFront VPC origin group.
  It introduces no useful extra Internet egress (explicit loopback-only egress).
- Cache policy with minimum/default/maximum TTL0; supported error TTLs also0.
  Viewer Host/Origin/CSRF headers and cookies reach the app; JSON POST is enabled.
  There is no distribution access logging configuration containing judge cookies.

No template resource creates an EC2 instance, model store, bucket, IAM user/key,
subnet or route table, or attaches/replaces an existing network interface group set.
CloudFront can require its standard AWS service-linked role; do not grant custom
cross-account access or create credentials as a workaround for denied permissions.

## Required real-account preflight

Sign into the already authorized AWS account. No password, MFA code, browser
session token or access key belongs in GitHub, the deployment plan or chat.

Before creating even a disabled stack, inspect the existing instance and VPC:

1. Confirm account, instance ARN, region, ARM64 instance, running state, private
   DNS/interface and the currently attached security groups.
2. Confirm VPC-origin support for the exact region/AZ, subnet IPv4 capacity,
   Internet gateway prerequisite, route topology and network ACL return traffic.
3. Review the **union of every existing security group's rules**. Adding a
   restrictive group does not cancel a broad rule in another group. TCP8780 must
   not already be permitted from arbitrary Internet sources. If it is, redesign
   the isolated ingress before publishing; do not silently alter shared groups.
4. Confirm the instance/subnet is eligible for a VPC origin. This is not established
   by knowing that EC2 has a private IP. Do not convert a shared subnet, remove the
   public address, move the workload or break SSH merely to fit this design.
5. Preserve the actual interface group set and deployment state for rollback.
   Verify the operator's access and AWS permission scope before applying changes.

A failed eligibility check leaves the currently working private portal in place.
Do not fall back to public HTTP carrying credentials or to a self-signed browser
warning. Repeat this preflight from an authenticated AWS session for each deployment.

## Staged deployment after preflight

Use a new stack name and the generated template. The following environment values
are supplied privately from verified discovery; they are not repository defaults:

```sh
aws cloudformation deploy --region us-east-1 --stack-name care-vision-judge-https --template-file /PRIVATE/new-https-plan/cloudfront-vpc-origin.json --parameter-overrides ExistingInstanceArn="$JUDGE_INSTANCE_ARN" ExistingVpcId="$JUDGE_VPC_ID" ExistingPrivateDns="$JUDGE_PRIVATE_DNS"
```

At this point the distribution is still disabled and the dedicated portal group
has no ingress. VPC-origin provisioning can take several minutes. Inspect actual
stack events; do not infer success from a submitted create request.

Read the stack outputs. Find the AWS-created `CloudFront-VPCOrigins-Service-SG`
in the **same VPC/account**, verify its provenance, and supply its actual ID.
Never create, edit or rename a group using AWS's reserved service-group name.

Review the primary network interface snapshot, then append the new portal group
while preserving every original group. Do not execute a replacement group list
assembled from stale console state. The template itself does not perform this
attachment. Keep the original instance and unrelated ingress policies intact.

Start a separate supervised portal on the instance's verified RFC1918 interface,
with a new private state directory and the exact distribution viewer origin:

```sh
PYTHONPATH=/DEPLOY/src /DEPLOY/.venv/bin/python -m care_vision.judge --credentials-file /PRIVATE/access.json --state-root /PRIVATE/https-judge-sessions --cloud-python /DEPLOY/.venv/bin/python --cloud-source /DEPLOY --private-bind "$JUDGE_PRIVATE_IPV4" --https-origin "https://$JUDGE_CF_DOMAIN" --port 8780 --seconds 120
```

The private bind is explicit, requires an HTTPS viewer origin and rejects wildcard,
public, loopback and metadata addresses. It is **not** a firewall substitute:
confirm the complete security-group union before binding a service on the VPC
interface. The regular default remains loopback-only, so existing SSH operation
is unaffected. An instance with a public NAT address still needs the network
restrictions; a private interface alone does not establish private reachability.

Keep credentials/state outside Git. Use owner-only permissions and a persistent
supervisor restricted to this deployment, with worker cleanup on stop. Do not
expose the cloud WebSocket port or the old evidence file server. HTTP on the VPC
origin is acceptable only over the verified CloudFront private origin connection;
never substitute an unrestricted custom public origin.

Only after the origin and group attachment are verified, update the stack with
the same instance/VPC parameters plus:

```text
ManagedCloudFrontSecurityGroup=<verified AWS-managed group ID>
EnableEndpoint=true
```

This enables only the reviewed login/application entry. Account/network changes
must match the user's authorized deployment scope. No judge credentials are
published in CloudFormation outputs or in Devpost.

## Acceptance before putting a URL in Devpost

Verify from a separate network/browser, without the SSH tunnel:

- Trusted HTTPS hostname/certificate and a deployed distribution; HTTP rejected.
- Anonymous API denied; viewer cannot start/stop or see the worker connection port.
- Operator start/stop/timeout and invalid-CSRF/Origin rejection.
- HttpOnly/Secure/SameSite session cookie, no cached cross-session state and no
  request/cookie logging introduced by the delivery path.
- No file traversal, recording/image/model/raw-log routes; no public reachability
  of the EC2 origin port.
- The actual intended person/pose workload and its declared sensing profile.
  Do not substitute the earlier geometric motion probe for incident verification.
- Explicit stale/offline status and a controlled end-to-end rehearsal. The prior
  probe failed its requested20FPS rate audit; HTTPS does not resolve that result.

Local locked schema/security tests alone cannot establish these real CloudFront
checks. For a new deployment, leave Devpost's working endpoint empty until the
external entry is verified. Sharing a URL does not authorize sharing private
viewer credentials, judge invitations, recordings or real incident information.

## Rollback

Disable the newly created CloudFront distribution first. Stop only the new HTTPS
portal process/supervisor and its owned workers. Restore the snapshotted interface
group set by removing only the newly added portal group; preserve original groups
and SSH. Keep the previous loopback deployment and all experiment/evidence files.
Remove only the stack/resources created for this deployment when no longer needed;
check that they are detached, disabled and no longer in use. Do not delete input
recordings, models, original groups, instance, subnet or SDK.

## Primary references

- [CloudFront VPC origin prerequisites and managed security group](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-vpc-origins.html)
- [CloudFormation VPC origin schema](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-cloudfront-vpcorigin.html)
- [EC2 group defaults and restrictive egress](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-ec2-securitygroup.html)

## Initial staged-account result — 19 October 2026

Authenticated discovery confirmed the existing Arm instance, supported AZ, spare
subnet IPv4 addresses, Internet gateway, return-traffic ACL and complete attached
security-group union. The original group permits SSH but does not expose the
portal port. The existing subnet **is publicly routed**; no subnet conversion,
route, public address or original SSH group was changed. AWS reports the EC2 VPC
origin as deployed on this topology. This is a private origin connection to an
instance in a publicly routed subnet, not a claim that the subnet is private.

A disabled CloudFront distribution, cache/request policies and a dedicated group
with no ingress were created. The group has not been attached to the instance.
A separate systemd-supervised portal is active on the private interface, with the
exact assigned HTTPS viewer origin and a 120-second worker limit. The original
loopback portal remains separately bound. Enabling the reviewed external path
still requires the managed-source ingress rule, group attachment and distribution
update, followed by external acceptance. No working endpoint has been added to
Devpost from this staging result.

The isolated locked runtime verified OpenCV5.1.0-dev and COOL on Arm. The approved
person model already present on the instance matched its frozen SHA256; native
CLASSIC inference on a generated blank frame returned zero people. The portal now
configures that experimental person detector, without inherited pose/scene
calibration or a shortened TEST-alert threshold. This is **not** the complete
frozen person/pose evaluation profile or evidence of incident accuracy.

Private-origin API checks passed anonymous denial, viewer start denial and hidden
worker port, Host/Origin/CSRF guards, Secure/HttpOnly/SameSite cookies, no-store,
missing file/log/model routes, real phase3 COOL worker start and operator stop.
All 87 source/runtime-package files matched the staged manifest; no owned worker
remained and the three protected episode files checked in this run were unchanged.
No personal media or live camera was used. These checks do not establish external
HTTPS acceptance or resolve the prior 14-versus-20FPS probe failure.

## Activation and external acceptance — 19 October 2026

After the owner's specific activation confirmation, the template added TCP8780
from the verified AWS-managed CloudFront group. The portal group was appended to
the freshly read primary-interface group set, preserving the original SSH group.
The distribution was then enabled. CloudFormation finished UPDATE_COMPLETE and
CloudFront reported Deployed/Enabled. Its actual origin configuration uses the
VPC origin, its viewer policy is HTTPS-only, all cache TTLs are0, Host/Origin/
cookies are forwarded, and distribution access/cookie logging is disabled.

External browser/API checks passed the trusted HTTPS login page and JavaScript,
HTTP403, anonymous API401, invalid login denial, distinct viewer/operator roles,
Secure/HttpOnly/SameSite cookies, no-store, read-only viewer/no worker port,
Origin/CSRF rejection, absent private file routes, real phase3 COOL worker start,
explicitly absent live evidence without an Edge, native expiry, restart, explicit
stop and logout revocation. A direct connection to the instance's public8780
port timed out from the test client; the reviewed security-group union contains
no Internet grant for that port. This is scoped acceptance, not a comprehensive
security assessment or general-network resilience result.

The native worker stopped with demonstration_time_limit at120.0009 seconds.
It completed cleanly before the portal watchdog could label TIME_LIMIT. The
initial harness's stricter state assertion failed, so the native log was verified
and an external continuation completed restart/stop/logout checks. Two other
harness attempts are retained: system Python lacked a local issuer store (the
locked Python3.11 OS CA store validated TLS), and traversal was rejected400 by
CloudFront rather than reaching the origin's404. No TLS validation, firewall or
application control was weakened to pass these checks.

The working endpoint and credential-free guided testing instructions were saved
in Devpost and read back from the saved form. The submission remains a draft;
no final submission, terms acceptance, personal-media upload or judge invitation
was performed. Judge key/repository access delivery still requires authorized
recipients. The portal is a guided image-free interface; pose calibration, new
scenes and exact-profile rehearsal remain pending. The previous14-versus20FPS
result and COOL baseline limitations remain unchanged.
