import argparse
from dataclasses import dataclass
from peaceful_pie.unity_comms import UnityComms


@dataclass
class MyQuaternion:
    x: float
    y: float
    z: float
    w: float


def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port=args.port)
    res: MyQuaternion = unity_comms.getPosition(ResultClass=MyQuaternion)
    print("x:", res.x, "y:", res.y, "z:", res.z, "w:", res.w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    run(args)
