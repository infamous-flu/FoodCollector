import argparse
from dataclasses import dataclass
from typing import List
from peaceful_pie.unity_comms import UnityComms


@dataclass
class RayResults:
    rayDistances: List[List[float]]
    rayHitObjectTypes: List[List[int]]
    NumObjectTypes: int


@dataclass
class MyVector3:
    x: float
    y: float
    z: float


@dataclass
class RLResult:
    reward: float
    isFinished: bool
    observation: RayResults


def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port=args.port)
    res = unity_comms.reset(ResultClass=RayResults)
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    run(args)
