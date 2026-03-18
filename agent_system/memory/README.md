## Memory Manager

<p align="center">
    <img src="../../docs/gigpo/framework-comparison.png" alt="framework" width="100%">
</p>

`verl-agent` allows for flexibly choosing what history to include for each step, such as, recent steps, key events, summaries, or external knowledge.

We provide a simplest memory implementation as a starting point. Developers are encouraged to extend this module with custom memory strategies, such as dynamic summarization, selective memory retention, or external knowledge integration, to improve the handling of long-horizon interaction histories.

---

### How the Memory Module Works

The memory system is built around three components:

| Component | File | Role |
|---|---|---|
| `BaseMemory` | `base.py` | Abstract interface (`reset`, `store`, `fetch`) |
| `SimpleMemory` | `memory.py` | Default implementation with `[Obs N: '...', Action N: '...']` format |
| `ALFWorldMemory` | `memory.py` | ALFWorld-specific format with structured step blocks |

The `AlfWorldEnvironmentManager` (in `env_manager.py`) creates a memory instance and calls it in `build_text_obs()`:

```python
# env_manager.py — AlfWorldEnvironmentManager

def __init__(self, envs, projection_f, config):
    self.memory = ALFWorldMemory()   # ← plug in any BaseMemory subclass here
    ...

def step(self, text_actions):
    ...
    self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})  # ← store each step
    ...

def build_text_obs(self, text_obs, admissible_actions, init=False):
    if not init and self.config.env.history_length > 0:
        memory_contexts, valid_lens = self.memory.fetch(   # ← format history for the prompt
            self.config.env.history_length,
            obs_key="text_obs",
            action_key="action")
    ...
```

At every step `build_text_obs()` calls `memory.fetch()` which returns a formatted history string that is inserted into the prompt template (`ALFWORLD_TEMPLATE`).

---

### Customizing the Memory for ALFWorld

To write your own memory strategy, subclass `BaseMemory` and implement the three abstract methods:

```python
from agent_system.memory.base import BaseMemory
from typing import List, Dict, Any, Tuple

class MyALFWorldMemory(BaseMemory):
    """Example: compact, action-only history."""

    def __init__(self):
        self._data = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size

    def store(self, record: Dict[str, List[Any]]):
        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in record})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        memory_contexts, valid_lengths = [], []
        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            start = len(self._data[env_idx]) - len(recent)
            lines = [
                f"Step {start + j + 1}:\n"
                f"  Observation: {rec[obs_key]}\n"
                f"  Action: {rec[action_key]}"
                for j, rec in enumerate(recent)
            ]
            memory_contexts.append("\n".join(lines))
            valid_lengths.append(len(recent))
        return memory_contexts, valid_lengths
```

Then swap it in `AlfWorldEnvironmentManager.__init__`:

```python
from agent_system.memory.memory import MyALFWorldMemory   # your custom class

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = MyALFWorldMemory()   # ← swap here
        super().__init__(envs, projection_f, config)
```

The `fetch()` return value is injected into `ALFWORLD_TEMPLATE` as `{action_history}`, so any formatting change in `fetch()` is immediately reflected in what the model sees.

---

### Built-in Memory Implementations

| Class | Format | Best for |
|---|---|---|
| `SimpleMemory` | `[Observation N: '...', Action N: '...']` | General-purpose environments |
| `ALFWorldMemory` | Structured `Step N: / Observation: / Action:` blocks | ALFWorld (default) |
| `SearchMemory` | `Step N: {action} {observation}` | Search/retrieval tasks |

Developers are encouraged to add strategies such as **dynamic summarisation**, **selective retention** (keep only steps where reward changed), or **external-knowledge injection** for long-horizon tasks.
