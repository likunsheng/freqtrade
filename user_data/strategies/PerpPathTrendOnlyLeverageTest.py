from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from PerpPathTrendOnlyV1 import PerpPathTrendOnlyV1


class _PerpPathTrendOnlyLeverageBase(PerpPathTrendOnlyV1):
    leverage_value = 1.0

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return min(float(self.leverage_value), float(max_leverage))


class PerpPathTrendOnlyV1Lev1(_PerpPathTrendOnlyLeverageBase):
    leverage_value = 1.0


class PerpPathTrendOnlyV1Lev2(_PerpPathTrendOnlyLeverageBase):
    leverage_value = 2.0


class PerpPathTrendOnlyV1Lev3(_PerpPathTrendOnlyLeverageBase):
    leverage_value = 3.0


class PerpPathTrendOnlyV1Lev5(_PerpPathTrendOnlyLeverageBase):
    leverage_value = 5.0
