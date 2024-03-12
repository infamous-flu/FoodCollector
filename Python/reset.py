import argparse
from dataclasses import dataclass
from typing import List
import numpy as np
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


def _ray_results_to_np_array(ray_results: RayResults):
    distances_np = np.array(ray_results.rayDistances)
    distances_np = 1 / distances_np
    object_types_np = np.array(ray_results.rayHitObjectTypes)

    _obs = np.zeros(
        (ray_results.NumObjectTypes + 1, *distances_np.shape), dtype=np.float32
    )
    np.put_along_axis(
        _obs,
        np.expand_dims(object_types_np, axis=0),
        np.expand_dims(distances_np, axis=0),
        axis=0,
    )

    _obs = _obs[:-1]
    return _obs


def run(args: argparse.Namespace) -> None:
    unity_comms = UnityComms(port=args.port)
    res = unity_comms.reset(ResultClass=RayResults)
    print(_ray_results_to_np_array(res.observation))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    run(args)
