# Corrective Takeover Specification

## 1. Current State Assessment

The existing corrective takeover flow is a fully coupled joint-space path.

Evidence chain:

| Layer | File | Behavior |
|-------|------|----------|
| Teleop output | `so_leader.py` L155-161, `piper_leader.py` L407-425 | `get_action()` returns `{joint}.pos` absolute positions |
| Processor | `processor/factory.py` L27-35 | `IdentityProcessorStep` — no transform |
| Recording loop | `recording_loop.py` L306-310 | During intervention, teleop action passes through identity processor to `robot.send_action()` |
| Policy sync | `recording_hil.py` L196-206 | `PolicySyncDualArmExecutor.send_action()` sends same joint dict to both `robot.send_action()` and `teleop.send_feedback()` |
| Feedback | `so_leader.py` L163-171, `piper_leader.py` L428-453 | `send_feedback()` writes `{joint}.pos` as motor goal positions |

Conclusion: current takeover is pure absolute joint-space coupling with no FK/IK involvement.


## 2. Target Requirement

Support two coupling modes for human takeover, selectable at runtime via config:

### Mode A: `joint_coupled` (current behavior)

After the takeover button is pressed, the master arm and slave arm synchronize all joint states directly. The slave mirrors the master's absolute joint positions.

### Mode B: `ee_delta_ik` (new)

After the takeover button is pressed, the slave does NOT copy the master's absolute joint positions. Instead:

1. At takeover entry, latch both arms' EE poses via FK as reference points.
2. Each frame, compute the master's EE displacement (translation + rotation) relative to the latched reference.
3. Apply that relative displacement to the slave's latched reference to get a target EE pose.
4. Solve IK to convert the target EE pose to slave joint commands.
5. At takeover exit, clear all latched state.

This mode enables corrective intervention even when master and slave have different absolute configurations, as only the relative motion is transferred.


## 3. Target Hardware

Both modes must work on:

| Hardware | Leader URDF | Follower URDF | Notes |
|----------|------------|---------------|-------|
| SO100/101 | SO leader arm | SO follower arm | placo `RobotKinematics` already validated |
| Piper/PiperX | `piper_description/urdf/piper_no_gripper_description.urdf` | Same URDF | pinocchio used for gravity comp; placo for FK/IK |

Bimanual variants (`bi_so_*`, `bi_piper_*`) should be supported after single-arm validation.


## 4. Existing Reusable Building Blocks

| Component | File | Purpose |
|-----------|------|---------|
| `RobotKinematics` | `src/lerobot/model/kinematics.py` | placo-based FK/IK solver |
| `EEReferenceAndDelta` | `src/lerobot/robots/so_follower/robot_kinematic_processor.py` | Latched reference + delta computation (core logic exists) |
| `EEBoundsAndSafety` | Same file | Workspace clipping + max-step safety |
| `InverseKinematicsEEToJoints` | Same file | EE pose to joint angles via IK |
| `ForwardKinematicsJointsToEE` | Same file | Joint angles to EE pose via FK |
| `GripperVelocityToJoint` | Same file | Gripper velocity to position conversion |
| `ExoskeletonIKHelper` | `src/lerobot/teleoperators/unitree_g1/exo_ik.py` | pinocchio-based IK reference |

Note: `EEReferenceAndDelta` implements `use_latched_reference` with rising-edge latch semantics, which is the core pattern needed for `ee_delta_ik`. However, it is designed for gamepad delta inputs, not for leader-arm FK delta extraction. The new `TakeoverMode` abstraction extracts FK-based deltas from the leader arm directly.


## 5. Architecture Design

### 5.1 TakeoverMode Abstraction

New file: `src/lerobot/scripts/takeover_modes.py`

```python
class TakeoverMode(abc.ABC):
    @abc.abstractmethod
    def on_enter(self, leader_action: RobotAction, follower_obs: RobotObservation) -> None: ...

    @abc.abstractmethod
    def compute_action(self, leader_action: RobotAction, follower_obs: RobotObservation) -> RobotAction: ...

    @abc.abstractmethod
    def on_exit(self) -> None: ...
```

Why a handler abstraction rather than processor pipeline: the mode has enter/exit lifecycle with latched state, which maps poorly to stateless processor steps. The handler wraps existing FK/IK utilities and manages their lifecycle.

### 5.2 JointCoupledTakeover

Trivial implementation preserving current behavior:

- `on_enter`: no-op
- `compute_action`: return `leader_action` directly (identity)
- `on_exit`: no-op

### 5.3 EEDeltaIKTakeover

Stateful implementation:

- Holds two `RobotKinematics` instances (leader FK, follower FK+IK)
- Internal state: `leader_ref_pose`, `follower_ref_pose`, `last_safe_q`

Lifecycle:

- `on_enter`:
  - FK on leader joints -> `leader_ref_pose`
  - FK on follower joints -> `follower_ref_pose`
  - Store `last_safe_q` = current follower joints

- `compute_action`:
  1. FK on current leader joints -> `leader_now`
  2. Translation delta: `delta_pos = leader_now[:3,3] - leader_ref[:3,3]`
  3. Rotation delta: `delta_rot = leader_now[:3,:3] @ leader_ref[:3,:3].T`
  4. Target EE: `target[:3,3] = follower_ref[:3,3] + delta_pos`; `target[:3,:3] = delta_rot @ follower_ref[:3,:3]`
  5. Apply workspace bounds clipping and max-step limiting
  6. IK solve with `last_safe_q` as initial guess
  7. On IK failure: hold `last_safe_q`, log warning
  8. On success: update `last_safe_q`
  9. Return joint dict + gripper passthrough

- `on_exit`: clear all latched state

### 5.4 Integration Points

| File | Change | Description |
|------|--------|-------------|
| `src/lerobot/scripts/takeover_modes.py` | **NEW** ~250 lines | `TakeoverMode` ABC + two implementations |
| `src/lerobot/scripts/recording_loop.py` | Modify ~30 lines | Accept `takeover_mode` param; call lifecycle hooks at S0->S1, S1 loop, S1->S2 transitions |
| `src/lerobot/scripts/lerobot_record.py` | Modify ~40 lines | Add `takeover_mode` config fields to `RecordConfig`; construct handler; pass to `record_loop` |
| `src/lerobot/scripts/lerobot_human_inloop_record.py` | Modify ~10 lines | Pass through takeover config |

Files NOT modified: `teleoperator.py`, `so_leader.py`, `piper_leader.py`, `piper_follower.py`, `recording_hil.py`, `robot_kinematic_processor.py`.

### 5.5 Configuration

New fields in `RecordConfig`:

```python
takeover_mode: str = "joint_coupled"           # "joint_coupled" | "ee_delta_ik"
takeover_leader_urdf: str | None = None        # URDF path for leader FK
takeover_follower_urdf: str | None = None      # URDF path for follower FK+IK
takeover_ee_frame: str = "gripper_frame_link"   # URDF end-effector frame
takeover_max_ee_step_m: float = 0.05            # Max Cartesian step per cycle
takeover_ee_bounds_min: list[float] | None = None  # Workspace lower bounds [x,y,z]
takeover_ee_bounds_max: list[float] | None = None  # Workspace upper bounds [x,y,z]
```


## 6. Recording Loop Integration

### 6.1 Intervention State Machine Changes

Current state machine (unchanged):

```
S0 (POLICY) --[toggle]--> S1 (ACTIVE) --[toggle]--> S2 (RELEASE) --> S0
```

New hooks injected at transitions:

```
S0 -> S1: takeover_mode.on_enter(leader_action, follower_obs)
           set_teleop_manual_control(True)

S1 loop:  raw_leader_action = teleop.get_action()
           action = takeover_mode.compute_action(raw_leader_action, follower_obs)
           # replaces direct use of act_processed_teleop

S1 -> S2: takeover_mode.on_exit()
           set_teleop_manual_control(False)
```

### 6.2 Behavioral Guarantees

- When `takeover_mode=None` or `JointCoupledTakeover`: behavior is byte-for-byte identical to current code.
- `compute_action` is called AFTER `teleop_action_processor` (identity by default), so the existing processor pipeline is preserved.
- Gripper always passes through in joint space in both modes.


## 7. Safety

### 7.1 Guards in EEDeltaIKTakeover

| Guard | Implementation |
|-------|---------------|
| Workspace clipping | Clip target EE position to `[bounds_min, bounds_max]` before IK |
| Max-step limiting | If `norm(delta_pos) > max_ee_step_m`, scale delta to limit |
| Deadband | Ignore deltas below 0.5mm to filter leader arm vibration |
| IK failure hold | On solver divergence, hold `last_safe_q` and log error |
| Zero-motion on enter | `on_enter` only latches references; no command is issued |

### 7.2 Micro-Motion Safety Test Matrix

All tests must pass before moderate motion is attempted:

| Test | Mode | Motion | Pass Criterion |
|------|------|--------|----------------|
| 1 | joint_coupled | Toggle enter/exit, no arm motion | No joint jump |
| 2 | ee_delta_ik | Toggle enter/exit, no arm motion | No EE jump |
| 3 | joint_coupled | Single joint +2 deg | Slave follows same joint |
| 4 | ee_delta_ik | Single axis +3mm X | Slave moves +3mm X |
| 5 | ee_delta_ik | Single axis +3mm Y | Slave moves +3mm Y |
| 6 | ee_delta_ik | Single axis +3mm Z | Slave moves +3mm Z |
| 7 | ee_delta_ik | Rotation +2 deg around Z | Slave rotates +2 deg around Z |
| 8 | ee_delta_ik | Gripper open/close pulse | Slave gripper responds |
| 9 | ee_delta_ik | 10mm square XY trajectory | Slave traces square |
| 10 | Both | Full policy->takeover->release cycle | No jump at any transition |

Repeat full matrix for each hardware (SO100/101, Piper/PiperX) independently.


## 8. Implementation Order

1. Create `takeover_modes.py` with `TakeoverMode` ABC + `JointCoupledTakeover` + `EEDeltaIKTakeover`
2. Add config fields to `RecordConfig`
3. Integrate into `recording_loop.py` (lifecycle hooks at state transitions)
4. Wire config through `lerobot_record.py` and `lerobot_human_inloop_record.py`
5. Run safety test matrix on SO100 single-arm
6. Run safety test matrix on Piper single-arm
7. Validate bimanual variants


## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| IK solver divergence / joint jump | HIGH | Max-step clipping + IK failure hold + `EEBoundsAndSafety` logic |
| Leader/follower FK convention mismatch | HIGH | Mandatory FK/IK roundtrip validation (`IK(FK(q)) ~ q`) before any real motion |
| Sign convention error in delta | MEDIUM | Single-axis micro-motion tests for each axis independently |
| Mode switch during motion causes jump | MEDIUM | `on_enter` only latches; no command spike on toggle alone |
| Piper placo URDF not pulled via LFS | MEDIUM | Check file size > 200 bytes at init; fail fast with clear error |
| Gripper handling inconsistency | LOW | Gripper passes through in joint space in both modes |
| placo dependency unavailable | LOW | `RobotKinematics` already guards with ImportError |


## 10. Testing Strategy

### Unit Tests

- `JointCoupledTakeover.compute_action` is identity
- `EEDeltaIKTakeover.on_enter` correctly latches poses
- `EEDeltaIKTakeover.compute_action` produces expected delta for synthetic FK inputs
- `EEDeltaIKTakeover.on_exit` clears state; subsequent `compute_action` raises
- Max-step clipping activates when delta exceeds threshold
- IK failure hold returns `last_safe_q`

### Integration Tests (mock robot)

- Simulate full intervention toggle cycle with `JointCoupledTakeover`
- Simulate full intervention toggle cycle with `EEDeltaIKTakeover`
- Verify zero-motion safety: toggling produces no action change
- Verify dataset recording contains correct `complementary_info` annotations

### Hardware Validation

- Per the micro-motion safety test matrix (Section 7.2)
- Performed independently on each target hardware


## 11. Acceptance Criteria

The requirement is complete when all of the following hold:

- `joint_coupled` mode preserves current behavior exactly
- `ee_delta_ik` mode is selectable via `--takeover_mode=ee_delta_ik` without code edits
- Both SO100/101 and Piper/PiperX are supported with appropriate URDF paths
- Entering `ee_delta_ik` takeover does NOT snap the follower to the leader's absolute pose
- Releasing takeover cleanly returns control to policy without a jump
- FK/IK roundtrip validation passes on each target hardware before enabling `ee_delta_ik`
- All micro-motion safety tests pass before moderate motion is attempted
- Bimanual variants work after single-arm validation
