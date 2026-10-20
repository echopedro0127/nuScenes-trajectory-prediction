# Causal Context Phase 1 Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to execute this plan inline,
> one task at a time, with a review checkpoint after each task. Steps use checkbox syntax.

**Goal:** 建立基于整数微秒的可用帧选择，以及与评估标签分离的只读回放观测接口。

**Architecture:** 新增 context schema/timeline 模块和旧 replay 的 adapter；本阶段不改变
原 evaluator、模型、训练或数据缓存。下一阶段的新模型只接受观测接口，标签仅交给
loss/evaluator。

**Tech Stack:** Python 3.12、dataclasses、NumPy、pytest；沿用仓库现有依赖。

---

## 执行准备

工作目录为 `.worktrees/causal-context-online`，分支为 `feature/causal-context-online`。
开始前阅读 [技术设计](../specs/2026-10-20-causal-context-online-design.md)。
本阶段只使用合成数据，旧 data/cache/outputs 不需复制。

从开发目录运行：

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
New-Item -ItemType Directory -Path outputs -Force | Out-Null
git branch --show-current
git status --short
```

每次 pytest 使用独占临时目录：

```powershell
$testRunPath = Join-Path (Join-Path (Get-Location) 'outputs') ('pytest-' + [guid]::NewGuid().ToString('N'))
```

基线：2026-10-20，246 passed。不要把该数值作为 P1 新增测试数。

## 文件清单

| 动作 | 文件 | 职责 |
| --- | --- | --- |
| 创建 | src/trajectory_prediction/context/__init__.py | 包入口，保持为空 |
| 创建 | src/trajectory_prediction/context/schema.py | FrameRef 和时间戳校验 |
| 创建 | src/trajectory_prediction/context/timeline.py | 纯函数选择可用帧 |
| 创建 | src/trajectory_prediction/online/observations.py | 只读观测/标签和 replay adapter |
| 创建 | tests/context/test_timeline.py | 时间、延迟、过旧、通道和排序测试 |
| 创建 | tests/online/test_observations.py | 未来不变性、标签边界和兼容性 |
| 修改 | docs/DEVELOPMENT_ROADMAP.md | 完成后记录 P1 验收与提交 |

本计划中的代码块是待实施参考，当前分支只包含本文档。

## Task 1：定义 FrameRef 与时间选择

- [ ] **Step 1：创建测试文件 `tests/context/test_timeline.py`。**

```python
import pytest

from trajectory_prediction.context.schema import FrameRef
from trajectory_prediction.context.timeline import latest_available


def frame(token, source, available, channel="CAM_FRONT"):
    return FrameRef(token, channel, source, available)


def test_availability_excludes_delayed_and_future_frames():
    frames = (
        frame("old", 100, 100),
        frame("delayed", 200, 400),
        frame("future", 301, 301),
    )
    selected = latest_available(frames, "CAM_FRONT", 300)
    assert selected.token == "old"


def test_inclusive_boundary_channel_and_staleness():
    frames = (frame("camera", 100, 200), frame("radar", 199, 199, "RADAR_FRONT"))
    assert latest_available(frames, "CAM_FRONT", 200, max_age_us=100).token == "camera"
    assert latest_available(frames, "CAM_FRONT", 201, max_age_us=100) is None
    assert latest_available(frames, "LIDAR_TOP", 200) is None


def test_order_does_not_change_latest_source_selection():
    frames = (frame("a", 100, 290), frame("b", 200, 250))
    assert latest_available(frames, "CAM_FRONT", 300).token == "b"
    assert latest_available(tuple(reversed(frames)), "CAM_FRONT", 300).token == "b"


@pytest.mark.parametrize("source,available", [(-1, 0), (100, 99), (True, 100), (1.5, 10)])
def test_frame_rejects_invalid_timestamps(source, available):
    with pytest.raises(ValueError):
        frame("bad", source, available)
```

- [ ] **Step 2：运行并确认缺少新模块导致失败。**

```powershell
& $python -m pytest tests/context/test_timeline.py -q --basetemp $testRunPath
```

- [ ] **Step 3：新增空 `context/__init__.py`，并实施 schema。**

`src/trajectory_prediction/context/schema.py`：

```python
from dataclasses import dataclass


def validate_timestamp_us(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("timestamp must be a nonnegative integer in microseconds")


@dataclass(frozen=True)
class FrameRef:
    token: str
    channel: str
    source_timestamp_us: int
    available_timestamp_us: int

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token:
            raise ValueError("frame token must be a nonempty string")
        if not isinstance(self.channel, str) or not self.channel:
            raise ValueError("frame channel must be a nonempty string")
        validate_timestamp_us(self.source_timestamp_us)
        validate_timestamp_us(self.available_timestamp_us)
        if self.available_timestamp_us < self.source_timestamp_us:
            raise ValueError("availability cannot precede acquisition")
```

`src/trajectory_prediction/context/timeline.py`：

```python
from collections.abc import Iterable

from trajectory_prediction.context.schema import FrameRef, validate_timestamp_us


def latest_available(
    frames: Iterable[FrameRef],
    channel: str,
    timestamp_us: int,
    *,
    max_age_us: int = 2_000_000,
) -> FrameRef | None:
    validate_timestamp_us(timestamp_us)
    validate_timestamp_us(max_age_us)
    if not isinstance(channel, str) or not channel:
        raise ValueError("channel must be a nonempty string")
    candidates = (
        item for item in frames
        if item.channel == channel
        and item.available_timestamp_us <= timestamp_us
        and timestamp_us - item.source_timestamp_us <= max_age_us
    )
    return max(candidates, key=lambda item: (item.source_timestamp_us, item.token), default=None)
```

- [ ] **Step 4：补充同 timestamp/token tie-break、空列表、空 token/channel、非法查询时间
  和非法 max_age 测试，运行同一聚焦测试直到通过。**
  预期：所有断言通过，latest 指按采集时间最新且已经可用，而非按到达时间最新。
- [ ] **Step 5：提交本任务。**

```powershell
git add src/trajectory_prediction/context tests/context/test_timeline.py
git commit -m "feat: add causal context frame selection"
```

## Task 2：隔离观测与评估标签

- [ ] **Step 1：创建 `tests/online/test_observations.py`，写以下行为测试。**

```python
from dataclasses import fields

import numpy as np
import pytest

from trajectory_prediction.context.schema import FrameRef
from trajectory_prediction.online.observations import replay_observations
from trajectory_prediction.online.replay import replay_updates


def inputs():
    history = np.zeros((5, 2), dtype=np.float32)
    future = np.arange(24, dtype=np.float32).reshape(12, 2)
    return history, future


def test_future_mutation_cannot_change_current_observation():
    history, future = inputs()
    changed = future.copy()
    changed[1:] += 10000
    first, _ = next(replay_observations(history, future, anchor_timestamp_us=1_000_000))
    second, _ = next(replay_observations(history, changed, anchor_timestamp_us=1_000_000))
    np.testing.assert_array_equal(first.observed_history, second.observed_history)
    np.testing.assert_array_equal(first.observed_position, second.observed_position)
    assert first.timestamp_us == 1_500_000
    assert all("target" not in field.name and "future" not in field.name for field in fields(first))
    with pytest.raises(ValueError, match="read-only"):
        first.observed_history[0, 0] = 1


def test_adapter_preserves_old_replay_and_separates_labels():
    history, future = inputs()
    old = list(replay_updates(history, future))
    new = list(replay_observations(history, future, anchor_timestamp_us=1_000_000))
    assert len(old) == len(new) == 11
    for original, (observed, label) in zip(old, new, strict=True):
        assert observed.update_step == original.update_step
        np.testing.assert_array_equal(observed.observed_history, original.observed_history)
        np.testing.assert_array_equal(label.remaining_target, original.remaining_target)


def test_new_frames_are_released_only_when_available():
    history, future = inputs()
    frames = (
        FrameRef("old", "CAM_FRONT", 1_400_000, 1_400_000),
        FrameRef("delayed", "CAM_FRONT", 1_490_000, 1_600_000),
        FrameRef("future", "CAM_FRONT", 2_100_000, 2_100_000),
    )
    observed, _ = next(replay_observations(
        history, future, anchor_timestamp_us=1_000_000, frames=frames,
    ))
    assert tuple(item.token for item in observed.frames) == ("old",)
```

- [ ] **Step 2：运行，预期因 observations 模块尚不存在而失败。**

```powershell
$testRunPath = Join-Path (Join-Path (Get-Location) 'outputs') ('pytest-' + [guid]::NewGuid().ToString('N'))
& $python -m pytest tests/online/test_observations.py -q --basetemp $testRunPath
```

- [ ] **Step 3：实施 `online/observations.py`。**

```python
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from trajectory_prediction.context.schema import FrameRef, validate_timestamp_us
from trajectory_prediction.context.timeline import latest_available
from trajectory_prediction.online.replay import replay_updates


def _snapshot(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("observation arrays must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservationView:
    update_step: int
    timestamp_us: int
    observed_history: np.ndarray
    observed_position: np.ndarray
    frames: tuple[FrameRef, ...]

    def __post_init__(self) -> None:
        validate_timestamp_us(self.timestamp_us)
        if type(self.update_step) is not int or not 1 <= self.update_step <= 11:
            raise ValueError("update step must be in [1, 11]")
        history = _snapshot(self.observed_history)
        position = _snapshot(self.observed_position)
        if history.ndim != 2 or history.shape[1] != 2 or len(history) == 0:
            raise ValueError("history must have shape [H >= 1, 2]")
        if position.shape != (2,):
            raise ValueError("position must have shape [2]")
        frames = tuple(self.frames)
        if any(item.available_timestamp_us > self.timestamp_us for item in frames):
            raise ValueError("observation cannot contain an unavailable frame")
        object.__setattr__(self, "observed_history", history)
        object.__setattr__(self, "observed_position", position)
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True)
class EvaluationLabel:
    remaining_target: np.ndarray

    def __post_init__(self) -> None:
        target = _snapshot(self.remaining_target)
        if target.ndim != 2 or target.shape[1] != 2 or len(target) == 0:
            raise ValueError("target must have shape [T >= 1, 2]")
        object.__setattr__(self, "remaining_target", target)


def replay_observations(
    initial_history: np.ndarray,
    future: np.ndarray,
    *,
    anchor_timestamp_us: int,
    history_steps: int = 5,
    frames: tuple[FrameRef, ...] = (),
) -> Iterator[tuple[ObservationView, EvaluationLabel]]:
    validate_timestamp_us(anchor_timestamp_us)
    if np.asarray(future).shape != (12, 2):
        raise ValueError("context replay requires exactly 12 future XY points")
    records = tuple(frames)
    channels = sorted({item.channel for item in records})
    for update in replay_updates(initial_history, future, history_steps):
        timestamp_us = anchor_timestamp_us + update.update_step * 500_000
        selected = tuple(
            item for channel in channels
            if (item := latest_available(records, channel, timestamp_us)) is not None
        )
        yield (
            ObservationView(
                update.update_step, timestamp_us, update.observed_history,
                update.observed_position, selected,
            ),
            EvaluationLabel(update.remaining_target),
        )
```

这是 latest-frame adapter；多 sweeps 聚合在 P5 单独实现。FrameRef 只引用 sensor token，
本阶段不会实际加载 sensor 文件。此 adapter 用于现有 12 点/6 秒实验，不扩展预测终点。

- [ ] **Step 4：补充 constructor 拒绝 unavailable frame、修改原输入数组不影响已生成
  snapshot、标签只读、空 history/非有限值拒绝、未来帧删除不影响当前输出等测试。**
  直接构造超过 11 步的 replay 也应显式拒绝；保留旧 replay 的通用行为。
  用一个仅接受 ObservationView 的 dummy predictor 记录收到的字段，验证 label 不传入。
- [ ] **Step 5：运行 observations、新 timeline 和旧 replay 测试。**

```powershell
$testRunPath = Join-Path (Join-Path (Get-Location) 'outputs') ('pytest-' + [guid]::NewGuid().ToString('N'))
& $python -m pytest tests/context/test_timeline.py tests/online/test_observations.py tests/online/test_replay.py -q --basetemp $testRunPath
```

预期：新旧测试通过，原 replay 数值不变。不要在本任务提前改训练数据或 model forward。

- [ ] **Step 6：提交本任务。**

```powershell
git add src/trajectory_prediction/online/observations.py tests/online/test_observations.py
git commit -m "feat: separate replay observations from evaluation labels"
```

## Task 3：阶段验收与交付

- [ ] **Step 1：运行完整验收。**

```powershell
$testRunPath = Join-Path (Join-Path (Get-Location) 'outputs') ('pytest-' + [guid]::NewGuid().ToString('N'))
& $python -m pytest -q --tb=short --basetemp $testRunPath
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
git diff --check
```

预期：完整测试无失败、Ruff 和公开仓库检查通过；样本/训练/metrics 文件不纳入提交。

- [ ] **Step 2：更新 `docs/DEVELOPMENT_ROADMAP.md`。**
  P1 状态改为完成，记录新增测试数、全量通过数、实际命令、commit 与已知限制；说明
  “观测 adapter 已可用，生产 context evaluator 接入在 P2”，避免把接口完成误写成
  多模态系统完成。
- [ ] **Step 3：提交文档并检查工作区。**

```powershell
git add docs/DEVELOPMENT_ROADMAP.md
git commit -m "docs: record causal context phase one verification"
git status --short
```

预期：worktree 干净。交付 P1 总结后，下阶段细化 P2 Map-only 计划。

## 本阶段完成定义

可用时间边界、严格剩余 horizon、不可写 snapshot、label 分离和旧 replay 回归通过；
没有真实 sensor I/O，没有新模型训练，没有发布新增性能结论。P2 再将已验证接口接入
实际上下文特征、模型调用和 evaluator。
