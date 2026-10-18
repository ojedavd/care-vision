import argparse

from care_vision.fixture import make_video

parser = argparse.ArgumentParser(description="Generate quiet/moving/quiet demo video")
parser.add_argument("output")
parser.add_argument("--static", action="store_true")
args = parser.parse_args()
print(make_video(args.output, static=args.static))
