# G05 DROID Policy Server — 接口契约

> client(`droid-franka-client`)与 server(本仓库 `scripts/serve_policy.py`)之间的**唯一**接口。
> 两边各放一份;任一边改协议,另一边据此更新。client 与 server **零 Python import 依赖**,只认本协议。

## 传输

- WebSocket，消息体 **msgpack**(numpy 数组用 msgpack-numpy 扩展打包)
- 地址:`ws://<POLICY_HOST>:<POLICY_PORT>`(默认 `127.0.0.1:8000`)
- 无压缩(`compression=None`),不限帧大小(`max_size=None`)

## 握手

连接建立后,server **主动推一帧 `metadata`**(msgpack dict),目前为 `{"action_steps": <int>}`
(server 每次推理产出的 action chunk 步数,默认 16)。client 读到即认为就绪。
> 建议(未来):metadata 内加 `protocol_version`,client 启动时校验,协议漂移可立即报错。

## 推理请求(client → server)

msgpack dict:

```
{
  "images": {
    "exterior_image":    uint8 [3, H, W],   # 第三视角(ZED-2i),CHW
    "wrist_image":       uint8 [3, H, W],   # 腕部(ZED-M),CHW
    "dummy_wrist_right": uint8 [3, 224, 224],# 占位,固定零张量
  },
  "state": {
    "right_arm":     float32 [7],   # 关节角(rad)
    "right_gripper": float32 [1],   # = 1.0 - gripper_position
  },
  "task":            <str>,         # 指令
  "frequency":       <int>,         # DROID 控制频率(=15)
  "embodiment_type": <str>,         # 路由用,DROID 为 Droid_Franka
}
```

> **Chunk 缓存**:server 用 action chunk,响应里带 `need_obs`。当上一帧响应 `need_obs=False` 时,
> client **本步发空 dict `{}`**(server 复用缓存的 chunk,不需要新 obs);`need_obs=True` 时才发完整 obs。

## 推理响应(server → client)

```
{
  "action": {                       # 单步动作 dict(注意是 "action" 单数)
    "right_arm":     float [7],     # 绝对关节角(rad);别名 "joint_position" 亦可
    "right_gripper": float [1],     # 夹爪;别名 "gripper" 亦可
    ...                             # 可能含 "left_gripper" 等其它部位,client 只取 right_*
  },
  "need_obs": <bool>,               # 下一步是否需要 client 发新 obs(见上 Chunk 缓存)
  "cot_text": <str>,                # 可选:CoT 推理文本(BBox / Subtask / Action token)
}
```
或错误:
```
{ "error": { "code": <int>, "message": <str> } }
```

## Reset

```
client → { "__reset__": true }
server → { "__reset__": true }      # 确认
```

## 对应实现

- client 发送端:`droid-franka-client/eval/main.py` 的 `GalaxeaPolicyClient` + `_build_galaxea_raw_obs`
- server 接收端:本仓库 `scripts/serve_policy.py`
