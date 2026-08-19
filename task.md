# Kronos System Wiring — Tasks

- [x] Create implementation plan
- [x] Fix `model/kronos.py` relative import (remove sys.path hack)
- [x] Fix `kronos_engine/forecaster.py` vendor path injection (prepend + absolute)
- [x] Add `vendor_path` to `config.yaml`
- [x] Populate `kronos_engine/__init__.py` exports
- [x] Populate `data/__init__.py` exports
- [x] Populate `features/__init__.py` exports
- [x] Populate `regime/__init__.py` exports
- [x] Populate `signals/__init__.py` exports
- [x] Populate `risk/__init__.py` exports
- [x] Populate `backtest/__init__.py` exports
- [x] Populate `execution/__init__.py` exports
- [x] Fix `backtest/engine.py` — use `to_kronos_frame()`
- [x] Fix `main.py` — pass all config keys to `KronosForecaster`
- [x] Update `requirements.txt`
- [x] Verify: import sanity check
- [x] Verify: `python run_demo.py`
- [x] Verify: `python main.py backtest --mock`
