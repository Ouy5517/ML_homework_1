# 数据说明

本项目使用 **UCI Seoul Bike Sharing Demand** 数据集（8,760 条逐小时记录）。原始字段包含日期、小时、租借量、温度、湿度、降雨量、降雪量、节假日和系统运行状态等。

## 获取方式

运行训练命令时，如果 `data/SeoulBikeData.csv` 不存在，脚本会从 UCI 官方地址自动下载并解压：

<https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip>

也可以手动下载后把 CSV 放到本目录。提交时应同时保留本说明，以便在另一台电脑上复现。

## 本项目使用的字段与单位

| 字段 | 含义 | 单位/取值 |
|---|---|---|
| `date` | 观测日期 | `YYYY-MM-DD` |
| `hour` | 小时 | 0–23 |
| `weekday` | 日期对应星期 | Monday–Sunday，由日期派生 |
| `month` | 月份 | 1–12，由日期派生 |
| `temperature` | 温度 | °C |
| `humidity` | 相对湿度 | % |
| `rainfall` | 降雨量 | mm |
| `snowfall` | 降雪量 | cm |
| `holiday` | 是否节假日 | `Holiday` / `No Holiday` |
| `functioning_day` | 系统是否运行 | `Yes` / `No` |
| `rented_bike_count` | 预测目标：每小时租借量 | 辆/小时 |

训练前还会增加 `temperature_sq = temperature²`，用于比较加入温度二次项后对非线性关系的改善。

## 清洗规则

- 日期无法解析或必需字段为空的记录删除；
- 数值字段转换失败的记录删除；
- 租借量截断到不小于 0；
- 数据按时间顺序分为前 64% 训练集、中间 16% 验证集和最后 20% 测试集。验证集用于选择特征和模型超参数；确定方案后使用前 80% 重新训练，最后 20% 只用于最终评价，避免未来记录泄漏到训练过程。
