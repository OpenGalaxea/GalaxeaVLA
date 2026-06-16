from core.inference.factory import create_inference_engine
from core.communication.ros2_bridge import Ros2Bridge
from utils.message.message_convert import action_dict_to_robot_action
from scheduler.instruction.instruction import InstructionManager, InstructionAction
from std_msgs.msg import String
import time
import logging
import numpy as np
logger = logging.getLogger(__name__)


_GRIPPER_KEYS = ("left_gripper", "right_gripper")


def _binarize_gripper_inplace(action: dict, threshold: float) -> None:
    """In-place: gripper >= threshold → 1.0, else → 0.0. Missing keys ignored."""
    for k in _GRIPPER_KEYS:
        if k in action and action[k] is not None:
            print(action[k])
            action[k] = (np.asarray(action[k], dtype=np.float32) >= threshold).astype(np.float32)*100
    return action

class Scheduler:
    """Robot scheduler — sends obs to server, receives single-step action, publishes.

    The server (serve_policy) decides when to recompute vs serve from cached
    chunk (controlled by server's ``--action_steps``).  The client is
    intentionally simple: every tick it gathers obs, sends to server, receives
    one action step, and publishes it.
    """
    def __init__(self, config, recorder=None, binarize_gripper: bool = False, gripper_threshold: float = 0.5, visualize: bool = False):
        self.inference_engine = create_inference_engine(config)
        self.inference_engine.connect()
        self.ros2_bridge = Ros2Bridge(config)
        self.instruction_manager = InstructionManager(config["instruction"])
        self.ros2_bridge.register_subscription(String, 'hs/vlm_out2vla', self.instruction_manager._ehi_instruction_callback)
        self.cnt = 0
        self.embodiment_type = config['robot'].get('embodiment_type')
        self.step_freq = config['basic']['control_frequency']
        self.predict_action_ms_last: float | None = None
        self.predict_action_ms_ema: float | None = None
        self._predict_action_ema_alpha = 0.05
        self.recorder = recorder
        self.binarize_gripper = binarize_gripper
        self.gripper_threshold = gripper_threshold
        self.visualize = visualize
        if self.binarize_gripper:
            logger.info("gripper binarization enabled, threshold=%.3f", self.gripper_threshold)

    def run(self):
        """Main loop: gather obs → server → single action → publish → repeat.

        Uses a fixed-period timer so that the control frequency stays stable
        regardless of how long inference takes.  If inference overshoots the
        period, the next tick fires immediately (no accumulated drift).
        """
        period = 1.0 / self.step_freq
        next_tick = time.monotonic()

        while self.ros2_bridge.is_running():
            obs, result = self._step()

            if result is not None:
                if self.binarize_gripper:
                    _binarize_gripper_inplace(result['action'], self.gripper_threshold)
                action = action_dict_to_robot_action(
                    result['action'],
                    timestamp=self.ros2_bridge.now(),
                )
                self.ros2_bridge.publish_action(action)

                if self.visualize and obs is not None:
                    from utils.viz import show
                    show(obs.get("images", {}), result.get("cot_text"))

                # ---- Recorder (same interface as eval_open_loop) ----
                if self.recorder is not None:
                    self.recorder.add_step(
                        images=obs.get("images") if obs else None,
                        gt_action=None,  # no GT on real robot
                        pred_action=result["action"],
                        extra={k: v for k, v in result.items() if k not in ("action", "need_obs")},
                    )

            # Fixed-period wait: sleep until next tick, skip if already past
            next_tick += period
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Overshot — reset to avoid accumulated lag
                next_tick = time.monotonic()
            self.cnt += 1

        # Save any active recording on shutdown
        if self.recorder is not None and self.recorder.active:
            self.recorder.stop_and_save()

    def _step(self):
        """Gather obs → handle instructions → infer → return (obs, result).

        Returns (None, None) when no obs is available or instruction says skip.
        """
        obs = self.ros2_bridge.gather_obs()
        if obs is None:
            if self.cnt % 100 == 0:
                print("No observation")
            time.sleep(0.01)
            return None, None

        instruct_action = self.instruction_manager.get_instruction(obs)
        if instruct_action == InstructionAction.RESET:
            # Save recording before reset
            if self.recorder is not None and self.recorder.active:
                self.recorder.stop_and_save()
            # Notify server before local reset (triggers server-side episode save)
            try:
                self.inference_engine.predict_action(obs)  # obs["task"] == "reset"
            except Exception:
                logger.warning("Failed to send reset signal to server", exc_info=True)
            self.ros2_bridge.reset()
            return None, None
        elif instruct_action == InstructionAction.CONTINUE:
            # Start recording on new episode
            if self.recorder is not None and not self.recorder.active:
                self.recorder.start(obs.get("task", ""))
        elif instruct_action == InstructionAction.SKIP:
            return None, None

        # Attach embodiment_type for mixture-mode servers
        if self.embodiment_type:
            obs["embodiment_type"] = self.embodiment_type

        # Send raw obs, receive raw action dict from server
        t_start = time.perf_counter()
        result = self.inference_engine.predict_action(obs)
        t_end = time.perf_counter()

        predict_ms = (t_end - t_start) * 1000.0
        self.predict_action_ms_last = predict_ms
        if self.predict_action_ms_ema is None:
            self.predict_action_ms_ema = predict_ms
        else:
            a = self._predict_action_ema_alpha
            self.predict_action_ms_ema = (1 - a) * self.predict_action_ms_ema + a * predict_ms

        if self.cnt % 2 == 0:
            logger.info(
                "predict_action_timing(ms) last=%.2f ema=%.2f",
                predict_ms,
                (self.predict_action_ms_ema or 0.0),
            )
        return obs, result



if __name__ == "__main__":
    import toml
    config = toml.load("config.toml")
    scheduler = Scheduler(config)
    scheduler.run()
