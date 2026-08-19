# Kronos System Wiring — Tasks

- [x] Create implementation plan
- [ ] Fix `model/kronos.py` relative import (remove sys.path hack)
- [ ] Fix `kronos_engine/forecaster.py` vendor path injection (prepend + absolute)
- [ ] Add `vendor_path` to `config.yaml`
- [ ] Populate `kronos_engine/__init__.py` exports
- [ ] Populate `data/__init__.py` exports
- [ ] Populate `features/__init__.py` exports
- [ ] Populate `regime/__init__.py` exports
- [ ] Populate `signals/__init__.py` exports
- [ ] Populate `risk/__init__.py` exports
- [ ] Populate `backtest/__init__.py` exports
- [ ] Populate `execution/__init__.py` exports
- [ ] Fix `backtest/engine.py` — use `to_kronos_frame()`
- [ ] Fix `main.py` — pass all config keys to `KronosForecaster`
- [ ] Update `requirements.txt`
- [ ] Verify: import sanity check
- [ ] Verify: `python run_demo.py`
- [ ] Verify: `python main.py backtest --mock`
