#!/usr/bin/env python3
"""Debug script to diagnose SO100 small action issue.

Run this to check:
1. What units does lerobot robot return (degrees vs normalized)
2. What values does the server return
3. Whether RelativeJointTransform.backward is working correctly
"""

import sys
import numpy as np
from pathlib import Path

# Add the repository source tree to the import path.
DEPLOY = Path(__file__).parent
PROJECT = DEPLOY.parent.parent
sys.path.insert(0, str(PROJECT / "src"))

def check_lerobot_units():
    """Check what units lerobot SOFollower returns."""
    print("=" * 60)
    print("STEP 1: Check lerobot robot observation units")
    print("=" * 60)

    try:
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        print("✓ Found lerobot SOFollower")

        # Try to connect to robot
        try:
            config = SOFollowerRobotConfig(port="/dev/ttyACM0")  # Adjust port if needed
            robot = SOFollower(config)
            robot.connect()
            obs = robot.get_observation() if hasattr(robot, "get_observation") else robot.capture_observation()

            motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

            print("\nRobot observation values:")
            for m in motor_names:
                key = f"{m}.pos"
                if key in obs:
                    val = obs[key]
                    print(f"  {m:15s}: {val:8.3f}")

            # Check if values are in [-1, 1] range (normalized) or degrees
            vals = [obs[f"{m}.pos"] for m in motor_names if f"{m}.pos" in obs]
            if vals:
                min_val, max_val = min(vals), max(vals)
                print(f"\nValue range: [{min_val:.3f}, {max_val:.3f}]")

                if abs(min_val) <= 1.5 and abs(max_val) <= 1.5:
                    print("⚠️  WARNING: Values look NORMALIZED (in [-1, 1] range)")
                    print("   But training data expects DEGREES!")
                    print("   This is likely the root cause of small actions.")
                elif max_val > 10:
                    print("✓ Values look like DEGREES (correct)")
                else:
                    print("? Unclear - values are in unusual range")

            robot.disconnect()

        except Exception as e:
            print(f"✗ Could not connect to robot: {e}")
            print("  (This is OK if robot is not connected)")

    except ImportError as e:
        print(f"✗ Could not import lerobot: {e}")


def check_dataset_stats():
    """Check training dataset statistics."""
    print("\n" + "=" * 60)
    print("STEP 2: Check training dataset statistics")
    print("=" * 60)

    import json
    stats_path = DEPLOY / "run" / "dataset_stats.json"

    if not stats_path.exists():
        print(f"✗ dataset_stats.json not found at {stats_path}")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    state = stats['so100']['state']['right_arm']
    action = stats['so100']['action']['right_arm']

    motor_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

    print("\nTraining STATE (absolute position) ranges:")
    for i, name in enumerate(motor_names):
        q01 = state['global_q01'][i]
        q99 = state['global_q99'][i]
        print(f"  {name:15s}: q01={q01:7.2f}°, q99={q99:7.2f}°, range={q99-q01:6.2f}°")

    print("\nTraining ACTION (relative delta) ranges:")
    for i, name in enumerate(motor_names):
        q01 = action['global_q01'][i]
        q99 = action['global_q99'][i]
        print(f"  {name:15s}: q01={q01:7.2f}°, q99={q99:7.2f}°, range={q99-q01:6.2f}°")

    print("\n✓ Training data uses DEGREES (not normalized)")


def simulate_normalization_mismatch():
    """Simulate what happens if lerobot returns normalized values."""
    print("\n" + "=" * 60)
    print("STEP 3: Simulate normalization mismatch")
    print("=" * 60)

    import json
    stats_path = DEPLOY / "run" / "dataset_stats.json"

    with open(stats_path) as f:
        stats = json.load(f)

    state_stats = stats['so100']['state']['right_arm']

    # Simulate: lerobot returns normalized value 0.5 for shoulder_lift
    lerobot_normalized = 0.5

    # Training expects degrees, so normalizer will treat 0.5 as degrees
    # and normalize it using state q01/q99
    q01 = state_stats['global_q01'][1]  # shoulder_lift
    q99 = state_stats['global_q99'][1]

    # Normalizer forward: x_norm = (x - q01) / (q99 - q01) * 2 - 1
    # If x = 0.5 degrees (but should be ~120 degrees):
    wrong_normalized = (lerobot_normalized - q01) / (q99 - q01) * 2 - 1

    # Correct: if lerobot returned 120 degrees:
    correct_degrees = 120.0
    correct_normalized = (correct_degrees - q01) / (q99 - q01) * 2 - 1

    print(f"\nExample: shoulder_lift (training q01={q01:.2f}°, q99={q99:.2f}°)")
    print(f"\nIf lerobot returns NORMALIZED 0.5 (but training expects degrees):")
    print(f"  Normalizer sees: 0.5 degrees (WRONG!)")
    print(f"  Normalizes to: {wrong_normalized:.4f}")
    print(f"  This is completely wrong!")
    print(f"\nIf lerobot returns DEGREES 120.0 (correct):")
    print(f"  Normalizer sees: 120.0 degrees")
    print(f"  Normalizes to: {correct_normalized:.4f}")
    print(f"  This is correct!")

    print("\n⚠️  If lerobot returns normalized [-1,1] values:")
    print("   → State normalization will be completely wrong")
    print("   → RelativeJointTransform.backward will add wrong state")
    print("   → Final absolute target will be wrong")
    print("   → Robot will barely move!")


def suggest_fix():
    """Suggest how to fix the issue."""
    print("\n" + "=" * 60)
    print("SUGGESTED FIX")
    print("=" * 60)

    print("""
If lerobot returns NORMALIZED values but training expects DEGREES:

Option 1: Convert lerobot output to degrees in client
---------------------------------------------------------
In so100_policy_client.py, after getting robot observation:

    # Get lerobot observation (normalized [-1, 1])
    state_normalized = np.array([robot_obs[f"{m}.pos"] for m in _MOTOR_NAMES])

    # Convert to degrees using robot's joint limits
    # (Get these from robot calibration file or lerobot config)
    joint_mins = np.array([-180, -90, -180, -180, -180, 0])  # example
    joint_maxs = np.array([180, 180, 180, 180, 180, 100])    # example

    state_degrees = (state_normalized + 1) / 2 * (joint_maxs - joint_mins) + joint_mins

    # Use state_degrees in raw_obs
    raw_obs = {
        "images": images,
        "state": {"right_arm": state_degrees},  # <-- degrees, not normalized
        ...
    }

Option 2: Check lerobot configuration
---------------------------------------------------------
Check if lerobot has a setting to return degrees instead of normalized values.
Look in robot config or calibration files.

Option 3: Scale all actions by a constant factor
---------------------------------------------------------
If the issue is just that actions are too conservative, you can scale them:

    absolute_target = np.asarray(response["action"]["right_arm"])

    # Compute delta from current state
    delta = absolute_target - state

    # Scale delta (try 2.0, 3.0, etc.)
    scaled_delta = delta * 2.0

    # Apply scaled delta
    scaled_target = state + scaled_delta

    robot.send_action(scaled_target)
""")


if __name__ == "__main__":
    check_lerobot_units()
    check_dataset_stats()
    simulate_normalization_mismatch()
    suggest_fix()

    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Run this script to see lerobot observation values")
    print("2. Add logging in so100_policy_client.py to print:")
    print("   - state values from robot")
    print("   - absolute_target from server")
    print("3. Check if values are in degrees or normalized range")
    print("4. Apply the appropriate fix above")
