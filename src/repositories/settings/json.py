from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.llm.config import SystemConfig


JsonObject = dict[str, Any]


class SettingsRepository:
    """基于 JSON 文件或内存快照的设置仓储。

    中文说明：
    这个仓储只负责配置数据的读取与保存，不负责配置校验、补默认值、
    也不负责拼装前端响应结构。这样服务层可以专注业务规则，仓储层专注 IO。
    """

    def __init__(
        self,
        path: str | Path | None = None,
        initial: JsonObject | None = None,
        system_path: str | Path | None = None,
        system: SystemConfig | JsonObject | None = None,
    ):
        """初始化设置仓储。"""

        self.path = Path(path) if path else None
        self.system_path = Path(system_path) if system_path else Path("config/system.yaml")
        self._memory = copy.deepcopy(initial or {})
        self._system = system if isinstance(system, SystemConfig) else SystemConfig.from_dict(system)

    def load(self) -> JsonObject:
        """读取当前配置快照。"""

        if self.path and self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return copy.deepcopy(self._memory)

    def save(self, data: JsonObject) -> None:
        """保存配置快照。"""

        normalized = copy.deepcopy(data)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 中文说明：先写同目录临时文件，再一次性替换正式配置。写入中断时，
            # 原配置仍可读取，避免模型设置只写了一半导致整个设置页打不开。
            descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(normalized, stream, ensure_ascii=False, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                if self.path.exists():
                    os.chmod(temporary, self.path.stat().st_mode & 0o777)
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        self._memory = normalized

    def system(self) -> SystemConfig:
        """读取系统级默认配置。"""

        if self.path:
            return SystemConfig.load(self.system_path)
        return self._system
