# Public episode review

Open [the three-episode data view](https://dqi6ba1tz68v1.cloudfront.net/trials/)
directly, or select **Episode evidence** on the
[CARE-Vision home page](https://dqi6ba1tz68v1.cloudfront.net/). No viewer key is
required. The four-perspective browser walkthrough remains on the home page;
operator-led technical telemetry is a secondary link in its footer.

## What to inspect

Select **Recovery**, **No recovery** or **Sitting**. Use **Play episode**, playback
speed, the source-time slider and milestone buttons to inspect the data. **Body
pose** and **Detection box** can be shown independently. Point and box coordinates
are synchronised to the same saved capture; no body motion is interpolated.

| Episode | Decision shown | Pose provenance |
| --- | --- | --- |
| Recovery | Recorded closure to `NORMAL` at 32.511182 s | 137 estimates calculated offline for presentation |
| No recovery | Recorded `ASSISTANCE_REQUIRED` test alert at 39.945067 s; no external notification | 137 estimates calculated offline for presentation |
| Sitting | `NORMAL` at 33.826 s in controller replay, alongside the original `UNRESOLVED` state | 101 saved cloud pose observations; 40 samples without requested pose remain absent |

The three trials contain 139, 140 and 141 saved cloud observations respectively.
Frame counts, transmission settings, processing times, applied ACK times and
controller states come from the selected historical trials. They are not new
measurements produced when a visitor opens the page.

Recovery/no-recovery pose was not recorded during those original trials. A
separate presentation pass used their archived AVI captures, the already logged
confirmed boxes, standard OpenCV 4.14.0 and hash-verified RTMPose-t Body7. The 274
local pose estimates are not original transmitted JPEG results, were not part of
the historical agent input and are not a Graviton/COOL benchmark. The displayed
cloud processing times exclude that later presentation work. Sitting reuses the
original cloud pose; its episode closure remains explicitly a controller replay.

## Public and private material

The public view contains four reviewed text assets: HTML, CSS, application
JavaScript and selected episode data. It contains no camera images, video or
audio. Only body keypoints 5–16 (shoulders through ankles) from the COCO layout
are exported; nose, eye and ear points are excluded. Joints outside the image or
with model response below 0.30 are omitted, and a link requires both endpoints.
This is a display filter, not a calibrated probability or an incident threshold.
Missing pose is not filled from another capture; pose over one second old is
hidden. Estimated joints can still be incorrect, especially under occlusion.

Original recordings and private evaluation inputs remain private. The separate
original trial viewer, its videos and ROI images still require an authenticated
session, including direct media requests. Publishing the data view does not grant
operator access or start a camera, cloud worker or external notification.

This presentation excludes household camera pixels; it does not claim complete
anonymisation, validate general fall-detection accuracy or establish physical seat
support. It is useful for reviewing the recorded policy and sensing changes, but
does not replace the original images for assessing computer-vision correctness.

The [approved proposal](approved-proposal.md) minimises unnecessary visual
transmission and exposure while keeping the primary vision workload on Graviton
with OpenCV 5 + COOL. Preparing this public data view does not change what images
AWS processed in the original trials, the Edge/cloud boundary or the controller.
