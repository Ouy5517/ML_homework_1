# RideCast：共享单车需求估计器

这是机器学习课程作业一的可运行项目。用户输入日期、小时和天气条件，网页调用 Flask 后端返回下一小时的预计租借量（辆/小时）。模型使用 UCI Seoul Bike Sharing Demand 数据集训练，训练与网页预测共用同一套 `scikit-learn Pipeline`。

## 在线预览

[打开 GitHub Pages 页面](https://ouy5517.github.io/ML_homework_1/)

GitHub Pages 只托管静态前端，因此在线预览在 Flask 未连接时会明确显示“前端演示估计”。要使用保存的真实 Pipeline，请按下文在本地启动 Flask 后端；后端接口和训练产物都已包含在仓库中。

## Render 云端部署

仓库根目录的 `render.yaml` 定义了一个 Render Python Web Service。连接本仓库并创建 Blueprint 后，Render 在构建时安装依赖、重新训练并保存模型，再由 Gunicorn 提供网页和 `/predict` 接口。部署完成后，请使用 Render 给出的 `onrender.com` 地址访问完整应用；GitHub Pages 地址仍是静态预览。

Render 免费实例可能在闲置后休眠，首次请求需要等待服务唤醒。健康检查地址为 `/health`。

## 项目结构

```text
backend/
  model.py          数据清洗、特征构造、Pipeline、评估与保存
  train_model.py    下载数据、训练模型、生成指标和图表
  app.py            Flask /predict 与 /health 接口
  __init__.py
data/
  README.md         数据来源、字段、单位与清洗规则
models/              训练后生成 ridecast_pipeline.joblib
reports/             训练后生成指标、系数、测试预测和散点图
index.html           前端表单与结果面板
app.js               表单校验、请求 /predict、结果动画
styles.css           页面样式
PRESENTATION.md      课堂 5 分钟展示提纲和讲稿
requirements.txt     Python 依赖
```

## 安装

建议 Python 3.10 或更高版本。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 训练模型

```powershell
python -m backend.train_model
```

第一次运行会从 UCI 官方地址下载 CSV；数据获取方式和字段单位见 [`data/README.md`](data/README.md)。训练完成后会生成：

- `models/ridecast_pipeline.joblib`：包含预处理和 Ridge 回归的完整 Pipeline；
- `reports/model_metrics.json`：时间顺序 80/20 测试集的 MAE、R²，以及不含温度二次项的基线；
- `reports/coefficients.csv`：模型系数，可用于课堂解释；
- `reports/test_predictions.csv`：测试集真实值与预测值；
- `reports/largest_error.json`：绝对误差最大的测试样本；
- `reports/actual_vs_predicted.png`：真实值与预测值散点图。

模型特征为 `hour`、由日期派生的 `weekday`/`month`、`temperature`、`temperature²`、`humidity`、`rainfall`、`snowfall`、`holiday` 和 `functioning_day`。数值特征先用训练集中的中位数填补缺失，再标准化；类别特征先用训练集众数填补，再 one-hot 编码。所有变换只在训练集拟合，测试集和网页请求复用同一 Pipeline。Ridge 系数表示“该特征增加一个标准差时，预测租借量改变多少辆/小时”；类别系数表示相对被省略基准类别的辆/小时差异。

## 启动后端

```powershell
python -m backend.app
```

浏览器访问 <http://127.0.0.1:5000/>。后端接口：

```http
POST /predict
Content-Type: application/json
```

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
{"prediction": 742}
```

`/health` 会返回模型是否已加载。缺少模型或输入越界时，接口返回明确错误；前端在后端未启动时保留“前端演示估计”提示，方便先验收页面。

## 只看前端

没有启动 Flask 时，也可以运行：

```powershell
python -m http.server 8765
```

再访问 <http://127.0.0.1:8765/>。这时页面会使用演示估计值，不代表训练模型结果。

## 测试

```powershell
python -m pytest -q
```

测试覆盖网页关键元素、无障碍错误提示、动效降级、特征派生、训练/预测特征一致性、输入校验和 `/predict` 响应。

## 适用范围与局限

模型只学习了数据集覆盖的首尔共享单车运营条件，适合课堂演示和相近条件下的小时级估计。极端天气、临时道路管制、车辆库存变化和节庆活动不在特征中；预测值应作为估计，不应直接当作运营承诺。测试集只用于最终评价，系数解释和模型选择以训练阶段的结果为依据。
