import argparse
from dataclasses import dataclass
from typing import List
from peaceful_pie.unity_comms import UnityComms


@dataclass
class MyVector3:
    x: float
    y: float
    z: float


@dataclass
class RayResults:
    rayDistances: List[List[float]]
    rayHitObjectTypes: List[List[int]]
    NumObjectTypes: int


@dataclass
class RLResult:
    reward: float
    finished: bool
    observation: RayResults


def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port=args.port)
    res = unity_comms.step(action=args.action, ResultClass=RLResult)
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--action", type=str, required=True)
    args = parser.parse_args()
    run(args)
