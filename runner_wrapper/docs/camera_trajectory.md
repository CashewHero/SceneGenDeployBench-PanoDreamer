# Camera Trajectory Inputs

`camera_trajectory` is file-backed. The runner receives a path under a sample in input; all trajectory camera defaults should live inside that YAML/JSON file.

Runner request fragment:

```json
{
  "inputs": {
    "data": {
      "sample-1": {
        "image": "/data/datasets/example/image/0001.png",
        "camera_trajectory": "/data/datasets/example/trajectories/sample_0001.yaml"
      }
    }
  }
}
```

Trajectory file example:

```yaml
convention: camera_to_world
coordinate_system: NED
units: meters
projection: equirectangular
resolution: [2560, 1280]
fov: [360, 180]
frames:
  - id: "000000_easy_left"
    index: 0
    frame: image/000000_easy_left.png
    timestamp: 0.0
    position: [0.0, 0.0, 0.0]
    rotation_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
```

File shape:

- `frames`: ordered list of camera states. Runners may process frames by list order.
- Each frame uses `position` plus `rotation_quaternion_xyzw`.
- Missing `position` defaults to `[0.0, 0.0, 0.0]`; missing `rotation_quaternion_xyzw` defaults to `[0.0, 0.0, 0.0, 1.0]`.
- Optional top-level fields: `units`, `resolution`, `fov`, `intrinsics`, `metadata`.
- Optional per-frame fields: `id`, `index`, `frame`, `timestamp`.

Supported values:

- `convention`: `camera_to_world`, `world_to_camera`
- `coordinate_system`: `NED`, `ENU`, `RDF`, `RUB`
- `units`: `meters`; if absent, units are unspecified
- `projection`: `equirectangular`, `pinhole`, `fisheye`, `orthographic`, `cubemap`

## Runner Handling

- Load the YAML/JSON file from the sample's `camera_trajectory` value under its input role.
- Read trajectory-level fields such as `convention`, `coordinate_system`, `units`, `projection`, `resolution`, and `fov` from the trajectory file when present.
- Missing `coordinate_system`, `units`, or `projection` or other fields means unspecified.
- Missing `position` defaults to zero; missing `rotation_quaternion_xyzw` defaults to identity.
- Runner wrappers should convert trajectory values as needed for the downstream model.
- Runners should tolerate unspecified fields unless they need them for conversion. Runners that require conversion should fail only when a missing or unknown value is actually needed.
