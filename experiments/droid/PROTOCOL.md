# G05 DROID Policy Server Interface Contract

> This is the only interface between the client (`droid-franka-client`) and this repository's server (`scripts/serve_policy.py`).
> Keep one copy on each side. If either side changes the protocol, update the other side from this document. The client and server have zero Python import dependency on each other and rely only on this protocol.

## Transport

- WebSocket with **msgpack** message bodies. NumPy arrays are packed with the msgpack-numpy extension.
- Address: `ws://<POLICY_HOST>:<POLICY_PORT>`; default is `127.0.0.1:8000`.
- No compression (`compression=None`) and no frame size limit (`max_size=None`).

## Handshake

After the connection is established, the server proactively sends one `metadata` frame as a msgpack dict. It currently has the form `{"action_steps": <int>}`.
This is the number of action chunk steps returned by each server inference call; the default is 16. The client treats the connection as ready after receiving it.

Future recommendation: add `protocol_version` to metadata and validate it at client startup so protocol drift fails immediately.

## Inference Request: Client To Server

msgpack dict:

```text
{
  "images": {
    "exterior_image":    uint8 [3, H, W],    # third-person view, ZED-2i, CHW
    "wrist_image":       uint8 [3, H, W],    # wrist view, ZED-M, CHW
    "dummy_wrist_right": uint8 [3, 224, 224],# placeholder fixed zero tensor
  },
  "state": {
    "right_arm":     float32 [7],   # joint angles in radians
    "right_gripper": float32 [1],   # 1.0 - gripper_position
  },
  "task":            <str>,         # instruction
  "frequency":       <int>,         # DROID control frequency, usually 15
  "embodiment_type": <str>,         # routing key; Droid_Franka for DROID
}
```

Chunk cache behavior: the server uses action chunks and includes `need_obs` in responses. If the previous response has `need_obs=False`, the client sends an empty dict `{}` for the current step; the server reuses the cached chunk and does not need a fresh observation. When `need_obs=True`, the client sends the full observation.

## Inference Response: Server To Client

```text
{
  "action": {                       # single-step action dict; note singular "action"
    "right_arm":     float [7],     # absolute joint angles in radians; alias "joint_position" is also allowed
    "right_gripper": float [1],     # gripper; alias "gripper" is also allowed
    ...                             # may contain left_gripper and other parts; the client consumes right_* only
  },
  "need_obs": <bool>,               # whether the next step needs a fresh client observation
  "cot_text": <str>,                # optional CoT inference text: bbox, subtask, or action token
}
```

Error response:

```text
{ "error": { "code": <int>, "message": <str> } }
```

## Reset

```text
client -> { "__reset__": true }
server -> { "__reset__": true }      # acknowledgement
```

## Implementations

- Client sender: `GalaxeaPolicyClient` and `_build_galaxea_raw_obs` in `droid-franka-client/eval/main.py`.
- Server receiver: this repository's `scripts/serve_policy.py`.
