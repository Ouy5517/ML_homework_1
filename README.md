# RideCast 前端页面

这是“共享单车需求估计器”的前端页面，视觉风格依据同目录的 `DESIGN (2).md`：深蓝宇宙背景、Blurple 主操作色、重字重标题、渐变功能卡片和沉浸式数据面板。

## 在线预览

[打开 GitHub Pages 站点](https://ouy5517.github.io/ML_homework_1/)

## 预览

直接双击 `index.html`，或在本目录运行：

```powershell
python -m http.server 8765
```

然后访问 `http://127.0.0.1:8765/`。

## 后端接口

表单会优先向 `POST /predict` 发送 JSON：

```json
{
  "date": "2026-09-22",
  "hour": 17,
  "weekday": "Tuesday",
  "month": 9,
  "temperature": 18,
  "humidity": 55,
  "rainfall": 0,
  "snowfall": 0,
  "holiday": "No Holiday",
  "functioning_day": "Yes"
}
```

后端返回格式：

```json
{"prediction": 742}
```

当 `/predict` 尚未接入时，页面会明确标注“前端演示估计”，便于先独立验收界面和交互。

## 测试

```powershell
python -m pytest tests/test_webpage.py -q
```
