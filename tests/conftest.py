# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_RUNTIME_CACHE = ROOT / ".pytest_runtime_cache"
_RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)

# Reduce noisy cache-directory warnings from matplotlib/fontconfig pulled by NetKet deps.
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_CACHE / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_CACHE))

Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
