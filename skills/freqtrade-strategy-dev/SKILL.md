---
name: freqtrade-strategy-dev
description: 用于本仓库（/workspaces/freqtrade）中基于 Freqtrade/FreqAI 的策略研发工作流：数据下载、回测、OOS 验证、hyperopt（含回撤约束）、以及常见报错（OOM/数据不够/频率限制）的排查与修复。用户希望“优化策略/降低回撤/跑回测与超参/上 dry-run 与实盘”时触发。
---

# Freqtrade 策略研发（本仓库工作流）

## 目标

- 在 `user_data/strategies/` 内迭代策略（规则或 FreqAI）。
- 用可复现的回测/超参流程衡量收益、回撤、月度稳定性。
- 避免常见踩坑：数据区间不覆盖、FreqAI 训练窗口不够、1m 多周期 OOM、配置键冲突。

## 本仓库关键文件

- 配置：
  - `user_data/config_bt_rule.json`：规则策略回测（无 FreqAI）。
  - `user_data/config_bt_freqai.json`：FreqAI 回测基础配置。
  - `user_data/config_bt_freqai_5m_mtf.json`：5m + 多周期（15m/1h）基线，适合低内存环境。
- 策略：
  - `user_data/strategies/PerpPathStrategy.py`：规则策略。
  - `user_data/strategies/PerpPathFreqaiStrategy.py`：FreqAI 策略（含阈值、过滤、time-stop、label）。
- Hyperopt loss：
  - `user_data/hyperopts/sample_hyperopt_loss.py`：支持离散回撤约束（5/8/10/12%），从 `hyperopt_loss_params` 读取。

## 标准流程（强制按顺序）

### 1) 数据覆盖检查（先检查再跑）

运行：

- `freqtrade list-data -c user_data/config_bt_freqai_5m_mtf.json --trading-mode futures --show-timerange`

规则：

- 任何回测/Hyperopt 的 `--timerange` 必须落在所有需要的 timeframes 的 `From/To` 覆盖区间内。
- FreqAI 需要“回测起点之前”的训练窗口数据（取决于 `train_period_days` + `startup_candle_count`）。

### 2) 下载数据（优先用 5m 基线避免 OOM）

下载 BTC/ETH futures 的 5m/15m/1h（覆盖训练缓冲 + 回测区间）：

- `freqtrade download-data -c user_data/config_bt_freqai_5m_mtf.json --trading-mode futures -p BTC/USDT:USDT ETH/USDT:USDT -t 5m 15m 1h --timerange 20241101-20260101 --prepend`

如果 `--prepend` 总是补不齐，使用“删掉重下”（更可靠）：

- `freqtrade download-data -c user_data/config_bt_freqai_5m_mtf.json --trading-mode futures -p BTC/USDT:USDT ETH/USDT:USDT -t 5m 15m 1h --timerange 20241101-20260101 --erase`
- 然后分别按 timeframe 重下（5m / 15m / 1h 各跑一次）。

#### 一键脚本（推荐）

本 skill 提供脚本（见 `scripts/`）用于减少手打与踩坑：

- 下载并验证覆盖：`skills/freqtrade-strategy-dev/scripts/wf_download.sh 20241101-20260101`
- 回测：`skills/freqtrade-strategy-dev/scripts/wf_backtest.sh 20250601-20260101 user_data/config_bt_freqai_5m_mtf.json PerpPathFreqaiStrategy XGBoostClassifier`
- Hyperopt：`skills/freqtrade-strategy-dev/scripts/wf_hyperopt.sh 20250601-20260101 150`
- Walk-forward（按月滚动回测）：`skills/freqtrade-strategy-dev/scripts/wf_walkforward.sh 20250601 20260101 30`

### 3) 回测（Backtesting）

规则策略：

- `freqtrade backtesting -c user_data/config_bt_rule.json -s PerpPathStrategy --timerange 20250101-20260101 --dry-run-wallet 1000 --fee 0.0004 --export trades --breakdown month`

FreqAI 策略：

- `freqtrade backtesting -c user_data/config_bt_freqai_5m_mtf.json -s PerpPathFreqaiStrategy --freqaimodel XGBoostClassifier --timerange 20250601-20260101 --dry-run-wallet 1000 --fee 0.0004 --export trades --breakdown month`

### 4) OOS 验证（必做）

至少再选一段不重叠的区间重复第 3 步；若 OOS 为负，优先改“结构”（退出/过滤/label），不要只调阈值。

### 5) Hyperopt（阈值稳健化 + 回撤约束）

配置约束写在 config 根级别：

- `"hyperopt_loss_params": { "max_drawdown_limit": "8%" }`

运行（单进程避免 OOM）：

- `freqtrade hyperopt -c user_data/config_bt_freqai_5m_mtf.json -s PerpPathFreqaiStrategy --freqaimodel XGBoostClassifier --timerange 20250601-20260101 --dry-run-wallet 1000 --fee 0.0004 --spaces buy sell -e 150 -j 1 --hyperopt-loss SampleHyperOptLoss --hyperopt-path user_data/hyperopts --min-trades 30`

产物：

- 超参结果：`user_data/hyperopt_results/*.fthypt`
- 参数导出：`user_data/strategies/PerpPathFreqaiStrategy.json`

## 常见错误与快速定位

- 直接 `Killed`：几乎都是 OOM（1m + 多周期 + 长区间最常见）。优先切到 `5m` 基线，减少 timeframes/periods/shift。
- `Found array with 0 sample(s)`：回测区间没有数据（`list-data --show-timerange` 先查）。
- `all training data dropped due to NaNs`：训练窗口不够（下载更早数据或缩短回测起点/调整训练窗）。
- `hyperopt_loss` 相关报错：参数请放 `hyperopt_loss_params`，不要用 `hyperopt_loss`（该键可能被用作 loss 类名字符串）。
