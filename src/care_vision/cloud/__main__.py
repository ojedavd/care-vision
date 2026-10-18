import argparse
import asyncio
import math
from functools import partial
from pathlib import Path

from care_vision.cloud.nanodet import NanoDetPersonDetector
from care_vision.cloud.server import CloudServer
from care_vision.cloud.yolox import YoloXPersonDetector, validate_dnn_engine
from care_vision.common.metrics import EventLog
from care_vision.common.resources import monitor_resources


def main():
    parser = argparse.ArgumentParser(description="CARE-Vision cloud-side service")
    parser.add_argument("--host", choices=["127.0.0.1", "::1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--run-seconds", type=float, help="optional bounded demonstration lifetime")
    parser.add_argument("--person-model", help="person detector ONNX model")
    parser.add_argument("--person-detector", choices=["nanodet", "yolox"], default="nanodet")
    parser.add_argument("--person-input-size", type=int)
    parser.add_argument("--person-threshold", type=float, default=0.35)
    parser.add_argument("--person-nms-threshold", type=float, default=0.3)
    parser.add_argument("--experimental-loss-5fps", action="store_true")
    parser.add_argument("--experimental-s-only-5fps", action="store_true")
    parser.add_argument("--experimental-weak-association", action="store_true")
    parser.add_argument("--experimental-yolox-classic-engine", action="store_true",
                        help="Opt-in classic YOLOX engine; requires phase3 S-only and SDK support")
    parser.add_argument("--experimental-alert-after-seconds", type=float)
    parser.add_argument("--experimental-episode-continuity", action="store_true")
    parser.add_argument("--episode-state-file", type=Path, help="Private pending episode metadata")
    parser.add_argument("--diagnostic-disconnect-after-episode-seconds", type=float)
    parser.add_argument("--verification-model", help="YOLOX-S ONNX model for loss verification")
    parser.add_argument(
        "--pose-shadow-config", type=Path, help="Optional passive cloud pose observer"
    )
    parser.add_argument("--experimental-pose-observation", action="store_true",
                        help="Opt-in bounded pose-driven ROI/full-frame cycle")
    args = parser.parse_args()
    if args.experimental_pose_observation and (
        not args.pose_shadow_config or not args.experimental_s_only_5fps or args.phase != 3
    ):
        parser.error("pose observation requires pose-shadow-config, S-only mode and phase 3")
    pose_shadow_factory = None
    if args.pose_shadow_config:
        if args.phase != 3:
            parser.error("pose shadow requires phase 3 / Graviton + COOL")
        from care_vision.cloud.pose_shadow import shadow_factory

        try:
            pose_shadow_factory = shadow_factory(args.pose_shadow_config)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"invalid pose shadow config: {error}")
    if args.run_seconds is not None and args.run_seconds <= 0:
        parser.error("run-seconds must be positive")
    if args.person_model and not Path(args.person_model).is_file():
        parser.error("person-model does not exist")
    if not 0 < args.person_threshold <= 1 or not 0 < args.person_nms_threshold <= 1:
        parser.error("person thresholds must be in (0, 1]")
    if args.person_input_size is not None and args.person_detector != "yolox":
        parser.error("person-input-size is only supported by yolox")
    if args.experimental_loss_5fps:
        if not args.person_model or args.person_detector != "yolox":
            parser.error("experimental loss verification requires a YOLOX-Nano person model")
        if args.person_input_size != 416:
            parser.error("experimental loss verification requires --person-input-size 416")
        if not args.verification_model or not Path(args.verification_model).is_file():
            parser.error("experimental loss verification requires a YOLOX-S verification model")
    elif args.verification_model:
        parser.error("verification-model requires --experimental-loss-5fps")
    if args.experimental_s_only_5fps:
        if args.experimental_loss_5fps:
            parser.error("S-only and loss-verification modes are mutually exclusive")
        if (
            not args.person_model
            or args.person_detector != "yolox"
            or args.person_input_size != 640
        ):
            parser.error("S-only mode requires a YOLOX-S model with --person-input-size 640")
    if args.experimental_yolox_classic_engine:
        if args.phase != 3 or not args.experimental_s_only_5fps:
            parser.error("classic YOLOX engine requires phase 3 and experimental S-only mode")
        try:
            validate_dnn_engine("classic")
        except ValueError as error:
            parser.error(str(error))
    from care_vision.cloud.controller import Policy

    try:
        Policy(
            experimental_s_only_5fps=args.experimental_s_only_5fps,
            experimental_alert_after_seconds=args.experimental_alert_after_seconds,
            experimental_episode_continuity=args.experimental_episode_continuity,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.episode_state_file and not args.experimental_episode_continuity:
        parser.error("episode-state-file requires --experimental-episode-continuity")
    value = args.diagnostic_disconnect_after_episode_seconds
    if value is not None and (
        not args.experimental_episode_continuity or not math.isfinite(value) or value <= 0
    ):
        parser.error("diagnostic disconnect requires continuity and a positive time")
    if args.experimental_weak_association and (
        not args.experimental_s_only_5fps or args.person_threshold != 0.35
    ):
        parser.error("weak association requires S-only mode and strong threshold 0.35")
    log = EventLog("cloud", args.log)
    detector_factory = None
    if args.person_model:
        detector_class = {
            "nanodet": NanoDetPersonDetector,
            "yolox": YoloXPersonDetector,
        }[args.person_detector]
        detector_options = {
            "score_threshold": 0.2 if args.experimental_weak_association else args.person_threshold,
            "nms_threshold": args.person_nms_threshold,
        }
        if args.person_input_size is not None:
            detector_options["input_size"] = args.person_input_size
        if args.experimental_yolox_classic_engine:
            detector_options["dnn_engine"] = "classic"
        detector_factory = partial(
            detector_class,
            args.person_model,
            **detector_options,
        )
        log.emit("person_detector_configuration", detector=args.person_detector,
                 dnn_engine_requested=("classic" if args.experimental_yolox_classic_engine
                                       else "auto"),
                 scope="person_detector_only; pose engine unchanged")
    verification_factory = (
        partial(
            YoloXPersonDetector,
            args.verification_model,
            input_size=640,
            score_threshold=0.35,
            nms_threshold=0.3,
        )
        if args.experimental_loss_5fps
        else None
    )

    async def run():
        async with monitor_resources(log):
            await run_service()

    async def run_service():
        service = CloudServer(
            log,
            detector_factory,
            verification_factory,
            args.experimental_loss_5fps,
            args.experimental_s_only_5fps,
            args.experimental_alert_after_seconds,
            args.experimental_weak_association,
            args.experimental_episode_continuity,
            args.episode_state_file,
            args.diagnostic_disconnect_after_episode_seconds,
            pose_shadow_factory=pose_shadow_factory,
            experimental_pose_observation=args.experimental_pose_observation,
        ).run(args.host, args.port, args.phase)
        if args.run_seconds is None:
            await service
        else:
            try:
                await asyncio.wait_for(service, args.run_seconds)
            except TimeoutError:
                log.emit("server_stopped", reason="demonstration_time_limit")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        log.close()


if __name__ == "__main__":
    main()
