# RideCast：共享单车需求估计器

RideCast 是机器学习课程作业一的可运行项目。用户输入日期、小时和天气条件，网页调用 Flask 后端，返回首尔共享单车每小时预计租借量（辆/小时）。训练和网页预测共用一条保存下来的 `scikit-learn Pipeline`，避免两端的特征处理不一致。

## 在线地址

| 用途 | 地址 |
|---|---|
| 手机/课堂展示入口（真实模型预测） | [打开 RideCast 网页](https://ridecast-ml-homework-1.onrender.com) |
| GitHub Pages 静态预览 | [ouy5517.github.io/ML_homework_1](https://ouy5517.github.io/ML_homework_1/) |
| GitHub 仓库 | [github.com/Ouy5517/ML_homework_1](https://github.com/Ouy5517/ML_homework_1) |
| Render 管理后台（需要登录） | [服务面板](https://dashboard.render.com/web/srv-dasepe60tbcc73f3mlg0) |

Render 服务当前跟踪 `main` 分支并自动部署。免费实例闲置后可能休眠，第一次访问需要等待唤醒。GitHub Pages 只能托管静态文件，后端不可用时会显示“前端演示估计”；要验证真实模型，请访问 Render 地址。

## 快速开始

### 本地运行完整应用

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m backend.train_model
python -m backend.app
```

打开 <http://127.0.0.1:5000/>。

### 只运行静态前端

```powershell
python -m http.server 8765
```

打开 <http://127.0.0.1:8765/>。未启动 Flask 时，页面会使用演示估计值，不代表训练模型结果。

## 项目流程

```mermaid
flowchart LR
    A[UCI Seoul Bike CSV] --> B[清洗与日期解析]
    B --> C[派生 weekday / month / temperature²]
    C --> D[按时间 64% / 16% / 20% 切分]
    D --> E[验证集选择特征与 alpha]
    E --> F[前 80% 重训 Pipeline]
    F --> G[中位数/众数填补]
    G --> H[标准化与 one-hot]
    H --> I[Ridge 回归]
    I --> J[joblib 模型]
    J --> K[Flask /predict]
    K --> L[网页结果]
```

## 数据与特征

数据集为 UCI [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)，包含 8,760 条逐小时记录。数据获取方式、字段单位和清洗规则见 [`data/README.md`](data/README.md)。训练脚本会在本地缺少 CSV 时自动下载官方压缩包。

预测目标：`Rented Bike Count`，单位为辆/小时。

输入特征：

- 时间：`date`、`hour`，以及由日期派生的 `weekday` 和 `month`；
- 天气：`temperature`（°C）、`humidity`（%）、`rainfall`（mm）、`snowfall`（cm）；
- 状态：`holiday`、`functioning_day`；
- 候选派生项：`temperature²`，用于检验温度与需求之间的非线性；是否采用由验证集决定。

数据按时间顺序分为前 64% 训练集、中间 16% 验证集和最后 20% 测试集。验证集只用于选择是否加入 `temperature²`、Ridge 的 `alpha` 和候选模型；确定方案后，用前 80%（训练集加验证集）重新拟合，最后 20% 只用于一次最终评价。数值字段用拟合数据的中位数填补并标准化，类别字段用拟合数据的众数填补后做 one-hot；所有变换只在对应训练数据上拟合。

## 训练与评估

```powershell
python -m backend.train_model
```

训练完成后会生成：

- `models/ridecast_pipeline.joblib`：预处理和 Ridge 回归组成的完整 Pipeline；
- `reports/model_metrics.json`：最终测试集 MAE、R²、验证集选择结果和 64%/16%/20% 切分信息；
- `reports/coefficients.csv`：模型系数；
- `reports/test_predictions.csv`：测试集真实值、预测值和误差分析数据；
- `reports/largest_error.json`：绝对误差最大的测试样本；
- `reports/actual_vs_predicted.png`：实际值—预测值散点图；
- `reports/prediction_curve.png`：固定其他输入后的小时条件预测曲线；
- `reports/prediction_heatmap.png`：小时 × 温度条件预测热力图；
- `reports/evaluation_dashboard.png`：最终测试集实际值—预测值散点图与残差图。

当前训练结果（最终测试集只评价一次）：

| 模型 | MAE（辆/小时） | R² |
|---|---:|---:|
| 验证集选择：Ridge 线性特征，alpha=0.1 | — | — |
| 最终测试：Ridge 线性特征，alpha=0.1 | **296.03** | **0.485** |

本次训练使用 5,606 条记录训练、1,402 条记录验证、1,752 条记录测试。验证集 MAE 为 520.58；该数值用于模型选择，不作为最终泛化性能报告。

## 模型函数

验证集选择了不含 `temperature²` 的 Ridge 线性模型，正则化参数为 `alpha=0.1`。数值特征先标准化，类别特征采用 one-hot 编码。模型在变换后特征空间中的函数为：

```text
ŷ_raw = -220.13
       + 191.99 z_hour + 6.29 z_month + 386.35 z_temperature
       - 118.99 z_humidity - 71.85 z_rainfall + 0.66 z_snowfall
       - 91.44 I(Monday) - 97.33 I(Saturday) - 163.79 I(Sunday)
       - 47.13 I(Thursday) - 50.85 I(Tuesday) - 35.10 I(Wednesday)
       + 156.92 I(No Holiday) + 839.90 I(Functioning Day=Yes)

ŷ = max(0, ŷ_raw)
```

其中 `z_feature = (feature - 训练/验证均值) / 训练/验证标准差`；`I(condition)` 为指示变量，条件成立时取 1，否则取 0。基准类别为星期五、`Holiday` 和 `Functioning Day=No`。由于数值特征已标准化，系数表示该特征增加一个标准差时对预测值的影响，而不是原始单位下的斜率。完整系数见 [`reports/coefficients.csv`](reports/coefficients.csv)。

## API

### 健康检查

```http
GET /health
```

模型已加载时返回：

```json
{"model_loaded": true, "status": "ok"}
```

### 预测

```http
POST /predict
Content-Type: application/json
```

请求示例：

```json
{
  "date": "2026-09-22",
  "hour": 17,
  "temperature": 18,
  "humidity": 55,
  "rainfall": 0,
  "snowfall": 0,
  "holiday": "No Holiday",
  "functioning_day": "Yes"
}
```

成功返回：

```json
{"prediction": 1057}
```

PowerShell 调用线上接口：

```powershell
$body = @{
  date = "2026-09-22"; hour = 17; temperature = 18; humidity = 55
  rainfall = 0; snowfall = 0; holiday = "No Holiday"; functioning_day = "Yes"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "https://ridecast-ml-homework-1.onrender.com/predict" `
  -Method Post -ContentType "application/json" -Body $body
```

缺少模型、缺少字段或输入超出范围时，后端会返回明确的错误状态和 `error` 字段。前端也会先做输入校验。

## Render 部署

仓库根目录的 [`render.yaml`](render.yaml) 保存了可复用的 Blueprint 配置：

- Python Web Service；
- 构建命令：`pip install -r requirements.txt && python -m backend.train_model`；
- 启动命令：`gunicorn backend.app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2`；
- `main` 分支自动部署；
- Python 版本固定为 3.12.8。

重新部署时，确保改动已推送到 `main`：

```powershell
git add .
git commit -m "update ridecast"
git push origin main
```

## 项目结构

```text
backend/
  model.py          数据清洗、特征构造、Pipeline、评估与保存
  train_model.py    下载数据、训练模型、生成指标和图表
  app.py            Flask /predict 与 /health 接口
scripts/
  generate_evaluation_charts.py  生成预测曲线、预测热力图和评估面板
data/
  README.md         数据来源、字段、单位与清洗规则
models/              训练后的 Pipeline
reports/             指标、系数、测试预测和模型评估图表
index.html           前端表单与结果面板
app.js               表单校验、请求 /predict、结果动画
styles.css           页面样式
requirements.txt     Python 依赖
render.yaml          Render Web Service 配置
```

## 测试

```powershell
python -m pytest -q
python -m compileall -q backend
node --check app.js
```

测试覆盖网页关键元素、无障碍错误提示、动效降级、空值校验、特征派生、训练/预测特征一致性、模型缺失状态和 `/predict` 响应。

## 适用范围与局限

模型学习的是首尔历史运营条件，适合课堂演示和相近条件下的小时级估计，不能直接代表其他城市、新站点或长期变化后的需求。大型活动、道路施工、站点库存、公共交通故障等变量没有纳入；极端天气属于训练范围外的外推。当前预测值用于估计，不应直接作为运营承诺。

## 课堂展示与图表

生成评估图表：

```powershell
python scripts/generate_evaluation_charts.py
```

最新课堂展示文件为 [`output/RideCast课堂展示-64-16-20-charts-v5.pptx`](output/RideCast课堂展示-64-16-20-charts-v5.pptx)。第 5 页包含测试集散点图和残差图，第 6 页包含预测曲线与小时 × 温度预测面，第 9 页说明完整模型函数和变量含义。
