# 项目分析报告 · Stock-Prediction-Models

> 本文档（RED）对 `Stock-Prediction-Models` 仓库进行系统性分析，涵盖项目概览、目录架构、技术栈、各模块详解、数据资产、核心实现剖析、技术债务与改进建议。

---

## 1. 项目概览

| 项目 | 说明 |
| --- | --- |
| 名称 | **Stock-Prediction-Models** |
| 作者 | huseinzol05（开源社区项目） |
| 许可证 | Apache License 2.0 |
| 定位 | 汇集**机器学习 / 深度学习模型**用于股票（及加密货币、外汇、黄金）价格预测，并包含**交易智能体**与**金融模拟**的综合性研究型仓库 |
| 形态 | 以 **Jupyter Notebook** 为核心交付物的研究/教学仓库，辅以少量 Python 模块与一个 Flask 实时服务 |
| 规模 | 约 **60+ Notebook**、30+ 深度学习模型变体、23 个交易智能体、5 个金融模拟、2 个集成模型 |

仓库是一个**算法展示与实验集合**，而非可投产的工程框架。每个 Notebook 通常自包含完整的「数据加载 → 预处理 → 建模 → 训练 → 可视化」流程，便于学习和复现单个算法。

---

## 2. 技术栈

### 2.1 语言与运行时
- **Python**（主体）+ **JavaScript**（TensorFlow.js 前端演示）
- 深度学习框架：**TensorFlow 1.x**（核心模型均基于 TF 1.x 图模式 API）

### 2.2 关键 Python 依赖

| 依赖 | 用途 | 出现位置 |
| --- | --- | --- |
| `tensorflow` (1.x) | 深度学习模型构建（`tf.placeholder`、`tf.nn.dynamic_rnn`、`tf.contrib.rnn`、`tf.train.AdamOptimizer`） | `deep-learning/`、`stacking/` |
| `sonnet` (DM Sonnet) | DNC（可微神经计算机）实现 | `deep-learning/dnc.py`、`access.py`、`addressing.py` |
| `numpy` / `pandas` | 数值计算与数据处理 | 全局 |
| `scikit-learn` | `MinMaxScaler` 归一化 | `realtime-agent/app.py` |
| `flask` | 实时交易智能体 HTTP 服务 | `realtime-agent/app.py` |
| `xgboost` / ARIMA | 集成堆叠模型 | `stacking/` |
| `pickle` | 模型序列化 | `realtime-agent/` |

### 2.3 前端（`stock-forecasting-js/`）
- 基于 **TensorFlow.js** 的浏览器端 LSTM / Signal-rolling 演示
- 含独立的 `css/`、`js/`、`fonts/`、`data/` 资源

> ⚠️ **重要**：项目未提供 `requirements.txt` 或 `setup.py`，依赖版本完全隐式。代码使用 TF 1.x 风格 API，无法直接在 TF 2.x 环境运行（详见第 7 节技术债务）。

---

## 3. 目录架构

```
Stock-Prediction-Models/
├── deep-learning/        # 18 个深度学习预测模型 + DNC 实现 + 工具
├── agent/                # 23 个交易智能体（强化学习 + 进化策略）
├── stacking/             # 2 个堆叠集成模型
├── simulation/           # 5 个金融模拟（蒙特卡洛 + 投资组合优化）
├── misc/                 # 数据探索与分析（TESLA/比特币/黄金等）
├── realtime-agent/       # 实时交易智能体（Flask 部署）+ 训练 Notebook
├── free-agent/           # 免费数据版演化策略智能体（含贝叶斯变体）
├── stock-forecasting-js/ # TensorFlow.js 浏览器端前端演示
├── dataset/              # 17 个 CSV 数据集
├── output/               # 深度学习模型结果图
├── output-agent/         # 智能体交易结果图
├── README.md             # 项目主文档（含全部模型清单与结果截图）
└── LICENSE               # Apache-2.0
```

---

## 4. 模块详解

### 4.1 `deep-learning/` — 深度学习预测模型（18 个 + 辅助）

按编号组织，覆盖主流时序深度学习架构：

| # | 模型 | 说明 |
| --- | --- | --- |
| 1-3 | LSTM / Bidirectional-LSTM / LSTM-2Path | 经典循环网络及其变体 |
| 4-6 | GRU / Bidirectional-GRU / GRU-2Path | 门控循环单元变体 |
| 7-9 | Vanilla / Bidirectional-Vanilla / Vanilla-2Path | 基础 RNN |
| 10-12 | LSTM Seq2seq / Bidirectional / VAE | 序列到序列 + 变分自编码器 |
| 13-15 | GRU Seq2seq / Bidirectional / VAE | GRU 版 Seq2seq |
| 16 | Attention-is-all-you-Need | Transformer 注意力机制 |
| 17-18 | CNN-Seq2seq / Dilated-CNN-Seq2seq | 卷积序列模型（含膨胀卷积） |

**Bonus Notebook**：`how-to-forecast.ipynb`（预测 t+N）、`sentiment-consensus.ipynb`（情绪共识预测）

**DNC 子系统**（源自 Google DeepMind，Apache-2.0）：`dnc.py`、`access.py`、`addressing.py`、`util.py` 构成完整的可微神经计算机（Differentiable Neural Computer）实现，基于 Sonnet 框架。

**辅助模块**：`autoencoder.py`（自编码器降维）

### 4.2 `agent/` — 交易智能体（23 个）

涵盖强化学习、进化策略、经典策略三大流派。**每个智能体每次仅交易 1 单位**。

| 类别 | 智能体 |
| --- | --- |
| 经典/规则 | Turtle-trading、Moving-average、Signal-rolling、ABCD strategy |
| 策略梯度 | Policy-gradient |
| 进化策略 | Evolution-strategy、Neuro-evolution、Neuro-evolution + Novelty search |
| Q-Learning 系 | Q-learning、Double、Recurrent、Double-Recurrent、Duel、Double-Duel、Duel-Recurrent、Double-Duel-Recurrent |
| Actor-Critic 系 | Actor-Critic、Duel、Recurrent、Duel-Recurrent |
| Curiosity 系 | Curiosity Q-learning、Recurrent-Curiosity、Duel-Curiosity |

### 4.3 `stacking/` — 集成堆叠模型（2 个）

1. `stack-rnn-arima-xgb.ipynb`：自编码器降维 + 深度 RNN + ARIMA + XGBoost 回归器
2. `stack-encoder-ensemble-xgb.ipynb`：Adaboost + Bagging + Extra Trees + Gradient Boosting + Random Forest + XGBoost

辅助模块：`model.py`（LSTM 封装）、`autoencoder.py`

### 4.4 `simulation/` — 金融模拟（5 个）

1. Simple Monte Carlo
2. Dynamic Volatility Monte Carlo
3. Drift Monte Carlo
4. Multivariate Drift Monte Carlo（BTC/USDT + Bitcurate 情绪）
5. Portfolio Optimization（投资组合优化，参考 pythonforfinance）

### 4.5 `misc/` — 数据探索与分析

| Notebook | 内容 |
| --- | --- |
| `tesla-study.ipynb` | TESLA 股票市场研究 |
| `outliers.ipynb` | K-means / SVM / Gaussian 异常值检测 |
| `overbought-oversold.ipynb` | 超买超卖分析 |
| `which-stock.ipynb` | 选股分析 |
| `fashion-forecasting.ipynb` | 时尚趋势预测（含交叉验证） |
| `bitcoin-analysis-lstm.ipynb` | 比特币 LSTM 分析 |
| `kijang-emas-bank-negara.ipynb` | 马来西亚金币价格 |

### 4.6 `realtime-agent/` — 实时交易服务

- `realtime-evolution-strategy.ipynb`：在多只股票（TWTR/GOOG/FB/LB 等 10 只）上训练演化策略
- `app.py`：基于 **Flask** 的实时交易 API 服务（端口 8005）
  - 加载预训练 `model.pkl`
  - 暴露端点：`/trade`、`/balance`、`/inventory`、`/queue`、`/reset`
  - 滑动窗口（window_size=20）状态构造 + Softmax 动作决策（买入/卖出/观望）
- `request.ipynb`：请求示例

### 4.7 `free-agent/` — 免费数据智能体

`evolution-strategy-agent.ipynb` 与 `evolution-strategy-bayesian-agent.ipynb`（贝叶斯优化变体）

### 4.8 `stock-forecasting-js/` — 浏览器端演示

独立的前端项目，将 LSTM 与 Signal-rolling 智能体用 TensorFlow.js 实现，支持动态上传 CSV，原部署于 `huseinhouse.com/stock-forecasting-js`。

---

## 5. 数据资产（`dataset/`）

共 17 个 CSV，覆盖多类金融标的：

| 类别 | 文件 |
| --- | --- |
| 美股 | AMD、FB、FSV、GOOG、GOOG-year、INFY、KNX、MONDY、MTDR、SINA、TMUS、TSLA、TWTR |
| 加密货币 | BTC-sentiment（含情绪） |
| 外汇 | eur-myr、usd-myr |
| 大宗 | oil（原油） |

`realtime-agent/` 与 `misc/` 内另含专用 CSV（如 AMD、CPRT、LYFT、fashion.csv、sentiment-bitcoin.csv）。

---

## 6. 核心实现剖析

### 6.1 深度学习预测范式（`deep-learning/`）

**训练/测试划分约定**：
- 训练集：起始时间 → 倒数第 30 天
- 测试集：最后 30 天
- 每个实验重复 10 次，以信号预测准确率（accuracy）为指标

README 记录的代表性准确率：LSTM 95.69%、Dilated-CNN-Seq2seq 95.86%、LSTM-Seq2seq-VAE 95.42%。

### 6.2 演化策略智能体（`realtime-agent/app.py`）

`app.py:43` 的 `Deep_Evolution_Strategy` 是核心算法：

- **无需梯度**：通过对权重注入高斯噪声（population）评估 reward，按 reward 标准化后更新权重（`app.py:81-89`）
- **奖励函数**（`app.py:255` `get_reward`）：`invests * 0.7 + score * 0.3`，综合平均投资回报率与总收益率
- **状态构造**（`app.py:25` `get_state`）：滑动窗口差分 + 相对偏移
- **动作空间**：0=观望 / 1=买入 / 2=卖出（`argmax` 决策）

### 6.3 堆叠 LSTM 模型（`stacking/model.py`）

`model.py` 定义了简洁的 LSTM 回归封装：`MultiRNNCell` + `DropoutWrapper` + `dynamic_rnn` + MSE 损失 + Adam 优化器，是 stacking Notebook 的共享组件。

### 6.4 DNC（可微神经计算机）

源自 DeepMind 的参考实现，由控制器（LSTM）+ 外部记忆（`access.MemoryAccess`）组成，`dnc.py:36` 的 `DNC` 类继承 `snt.RNNCore`，展示了高级记忆增强网络结构。

---

## 7. 技术债务与风险

| 优先级 | 问题 | 说明 |
| --- | --- | --- |
| 🔴 高 | **TensorFlow 1.x 过时** | 全部深度学习代码使用 `tf.placeholder`、`tf.contrib`、`tf.nn.dynamic_rnn`、`tf.train.AdamOptimizer` 等 1.x API，在 TF 2.x 下无法直接运行。`tf.contrib` 已被移除。需迁移至 `tf.keras` 或 `tf.compat.v1`。 |
| 🔴 高 | **无依赖管理** | 缺失 `requirements.txt` / `setup.py` / `environment.yml`，版本完全隐式，复现困难 |
| 🟠 中 | **无自动化测试** | 无单元测试、无 CI，纯 Notebook 形式，难以保证代码正确性与回归 |
| 🟠 中 | **实时服务安全风险** | `app.py:379` `/trade` 端点直接 `json.loads(request.args.get('data'))`，无输入校验；`app.run(host='0.0.0.0')` 对外暴露；`print(prob)` 调试残留 |
| 🟡 低 | **代码重复** | 各 Notebook 大量重复数据加载/预处理逻辑，未抽取为公共库 |
| 🟡 低 | **异常处理** | `app.py:205` 等处裸 `except:` 吞没异常 |
| 🟡 低 | **文档分散** | 仅有 README 与 realtime-agent/README，缺少模块级说明 |

---

## 8. 改进建议

### 短期（低成本、高收益）
1. **新增 `requirements.txt`**，固定关键版本（`tensorflow==1.15`、`dm-sonnet`、`scikit-learn`、`flask`、`xgboost` 等），或提供 `environment.yml`
2. **README 增加环境准备说明**：明确 Python 版本（建议 3.6）与 TF 1.x 要求
3. **修复 `app.py` 输入校验**：对 `/trade` 参数做空值/类型检查，避免 `None` 导致 `json.loads` 崩溃

### 中期
4. **TF 1.x → 2.x 迁移**：逐步将核心模型迁移至 `tf.keras`，或使用 `tf.compat.v1` 兼容模式 + 迁移脚本
5. **抽取公共工具层**：将各 Notebook 重复的数据加载、归一化、状态构造、评估指标逻辑封装为 `src/` 模块
6. **补充冒烟测试**：对 `stacking/model.py`、`realtime-agent` 的 `get_state`/`softmax`/`get_reward` 等纯函数加单元测试

### 长期
7. **结构化重构**：从「Notebook 集合」演进为「可安装 Python 包 + 示例 Notebook」的双层架构
8. **引入 CI**：GitHub Actions 跑 lint + 测试，保障演进质量
9. **数据与模型分离**：大体积 CSV 与 `model.pkl` 考虑用 Git LFS 或外部存储

---

## 9. 快速上手指南

```bash
# 1. 建议使用 Python 3.6 + TensorFlow 1.x 环境
#    （项目无 requirements.txt，以下为推断的最小依赖）
pip install "tensorflow==1.15" numpy pandas scikit-learn matplotlib jupyter

# 2. 启动深度学习预测 Notebook
jupyter notebook deep-learning/1.lstm.ipynb

# 3. 启动实时交易服务（需先在 realtime-agent/ 内训练生成 model.pkl）
cd realtime-agent
python app.py   # 访问 http://localhost:8005/

# 4. 请求示例
curl "http://localhost:8005/trade?data=[13.1, 13407500]"
```

---

## 10. 模型/模块速查索引

| 关注点 | 入口 |
| --- | --- |
| 基础 LSTM 预测 | `deep-learning/1.lstm.ipynb` |
| 最佳准确率模型 | `deep-learning/18.dilated-cnn-seq2seq.ipynb` (95.86%) |
| Transformer 注意力 | `deep-learning/16.attention-is-all-you-need.ipynb` |
| 演化策略智能体 | `agent/6.evolution-strategy-agent.ipynb` |
| 实时部署 | `realtime-agent/app.py` |
| 堆叠集成 | `stacking/stack-rnn-arima-xgb.ipynb` |
| 投资组合优化 | `simulation/portfolio-optimization.ipynb` |
| 浏览器演示 | `stock-forecasting-js/index.html` |
| DNC 实现 | `deep-learning/dnc.py` |

---

*生成时间：基于仓库当前 HEAD（commit `3326673`）。*
