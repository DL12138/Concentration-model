# 荧光浓度建模 APP（本地版）

一个**个人专用、仅在本机运行**的荧光试纸浓度检测与建模工具：上传试纸照片，应用自动完成「图像预处理 → 检测区定位 → 颜色特征提取 → 浓度预测/标定」的完整链路，最终输出浓度值 C ± 不确定度 U 及是否超限的判定。所有数据保存在本机，**不联网、不登录、无云数据库、无需部署**。

---

## 1. 项目解决什么问题

- **场景**：用固定装置（暗箱 / 固定支架 / 固定灯光）拍摄荧光试纸照片，一张图对应一条试纸，检测区位置在每张图之间大致固定。
- **检测原理**：试纸**没有 C/T 线**；同一检测区的**荧光颜色随浓度变化**（例如低浓度发绿、高浓度发红），应用依靠检测区的颜色特征定量。
- **解决的问题**：把「拍照 → 颜色特征 → 浓度值」的手工流程自动化——上传图片后全流程自动执行；每个自动步骤都可人工干预、修正；建立可复用的标准曲线（标定模型）后，日常检测直接调用；所有数据本地持久化，刷新、关闭、重启均不丢失，可备份、可恢复、可追溯。

## 2. 主要功能

| 模块 | 功能 |
| --- | --- |
| 首页总览 | 最近一次浓度结果（C±U）、生效标定模型的 R² 与 LOD、标定状态、今日/累计检测数、最近检测记录、快捷操作入口 |
| 检测工作流 | 一个页面内 5 步流水线：① 批量上传 → ② 预处理 → ③ ROI 分割 → ④ 特征提取 → ⑤ 结果。上传后自动依次执行，每步可展开手动调整，改后下游级联重算 |
| 标定建模 | 按已知浓度分组管理数据（每组多张重复图，显示均值±标准差）；拟合 **线性 / 二次多项式 / 指数 / 4PL** 四类模型并对比 **R² / RMSE / LOD**（4PL 为 4 参数模型，需要至少 4 个标定数据点，不足时该模型会提示数据不足而不参与对比）；剔除离群点后重拟合；曲线可视化；保存/激活/删除模型 |
| 检测记录 | 按批次、判定状态关键词检索历史结果；回看单条记录的完整参数快照（所用模型、特征值、标定数据、判定限）；导出 CSV |
| 数据与设置 | 判定上/下限、浓度单位、默认批次、默认滤波参数、暗场/平场参考图、ROI 模板管理、模型管理、一键备份 / 恢复 / 删除备份 |

**自动流水线**：上传后自动执行 预处理（默认参数）→ ROI（用激活模板自动定位）→ 特征提取 → 结果（检测图用生效模型输出 C±U；标定图标记为待加入标定数据集）。任一步骤异常或需人工时，图片标记为「需人工」（attention），可进入该图逐步修正；修正上游步骤后下游自动重算（级联）。

**检测判定**：浓度 C±U（95% 置信，k=2，由标定模型的预测区间估算）与设置页配置的判定上/下限比较，输出 `within`（正常）/ `above`（超上限）/ `below`（低于下限）/ `borderline`（临界）四种状态。

## 3. 安装方法

### 环境要求

- Windows 系统（开发与验证基于 Windows）
- Python 3.10 及以上（需能创建虚拟环境；本项目已在自带虚拟环境中验证）
- 依赖：`flask`、`numpy`、`opencv-python`、`scipy`（见 `fluro_app/requirements.txt`）

### 方式 A：使用项目自带虚拟环境（推荐）

项目已内置虚拟环境（`fluro_app\.venv\`，依赖已安装）。直接进入使用即可，见第 4 节。

### 方式 B：全新安装

```bat
cd fluro_app
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

安装完成后回到 `fluro_app` 目录双击 `start.bat`，或在命令行手动启动（见第 4 节）。

> 说明：`start.bat` 启动时优先使用项目内 `.venv\Scripts\python.exe`；若不存在则回退使用系统 Python。它**不会自动安装依赖**，全新环境请先按方式 B 安装。

## 4. 使用方法

### 启动

1. 双击 `fluro_app\start.bat`，或手动启动：

   ```bat
   cd fluro_app
   .venv\Scripts\python.exe server.py
   ```

2. 浏览器自动打开 `http://127.0.0.1:8000`（默认端口 8000；若被占用会自动在 8000~8019 之间选择空闲端口，也可用环境变量 `FLURO_PORT=9000` 指定）。
3. 关闭启动窗口即停止服务；数据已落盘，下次启动数据仍在。

### 首次使用（建立标定）

1. **上传标定图**：进入「检测工作流」→ 上传，导入已知浓度的荧光试纸照片，标记为「标定」并填写已知浓度，上传后自动流水线执行。
2. **保存 ROI 模板**：流水线第 ③ 步对首张图框选检测区，点击「存为模板」；之后的新图会自动套用模板（含模板匹配微调），也可单张手动修正。
3. **建模并保存生效模型**：进入「标定建模」，把标定图按浓度分组，拟合模型并对比 R² / RMSE / LOD，选择最优模型「保存为生效模型」。
4. 之后上传检测图，全流程自动执行并输出浓度 C±U 与超限判定，结果进入「检测记录」。

### 日常检测

上传检测图（可批量）→ 自动流水线出结果 → 首页总览与检测记录实时更新。

### 数据存储与备份

- **数据目录**：`fluro_app\data\`——`fluro.db`（SQLite 数据库：设置、批次、图片索引、流水线快照、ROI、特征、标定分组与数据点、模型、检测记录）+ `images\`（原图）+ `thumbnails\`（缩略图）+ `processed\`（处理图）+ `refs\`（参考图）。
- **备份**：在「数据与设置」页一键备份，备份打包为 `fluro_app\backups\backup_时间戳[_标签]\` 目录（含数据库与全部图片子目录），可随时恢复；恢复前会自动先备份当前数据。建议定期备份。

### 测试

```bat
cd fluro_app
.venv\Scripts\python.exe -m pytest tests/ -q
```

共 **94 项测试**（覆盖 M0~M9：环境与启动冒烟、数据层与上传、预处理、ROI 与特征、自动流水线、标定建模、检测判定、记录检索与导出、设置与备份恢复、验收）。

### 合成测试图工具

真实试纸图到位前，可用脚本生成模拟荧光图体验全流程：

```bat
cd fluro_app
.venv\Scripts\python.exe tools/make_test_images.py --out ../data/sample --conc 0,10,25,50,100 --reps 3
```

生成 800×600、暗色背景 + 居中圆形检测区、颜色随浓度从绿（0）到红（100）变化的图片，文件名含浓度信息，可直接导入应用。

## 5. 输入输出示例

### 输入

- **图片**：JPG / JPEG / PNG / BMP / TIF / TIFF / WEBP，一张图一条试纸；支持一次批量导入多张。
- **附加信息**：每张图标记「标定 / 检测」；标定图必须填写已知浓度（数字）；检测图可选填批次号。

### 输出

- **浓度**：C（浓度值，单位默认 ng/mL，可在设置页修改）
- **不确定度**：±U（95% 置信、k=2 的预测区间半宽，由标定模型估算）
- **判定**：within（正常）/ above（超上限）/ below（低于下限）/ borderline（临界，C±U 触及限值）

### 示例 1：上传 2 张标定图（浓度 10、25）并自动跑流水线

```
POST /api/images/upload   (multipart: files=图A.png, files=图B.png, kind=calibration, known_conc=10)
```

响应：

```json
{
  "ok": true,
  "images": [
    {"id": 1, "kind": "calibration", "batch": null, "known_conc": 10.0, "thumb_url": "/api/images/1/thumb"},
    {"id": 2, "kind": "calibration", "batch": null, "known_conc": 25.0, "thumb_url": "/api/images/2/thumb"}
  ],
  "errors": []
}
```

```
POST /api/pipeline/run   {"image_ids": [1, 2]}
```

响应（标定图第 ⑤ 步为 pending，待加入标定数据集）：

```json
{
  "ok": true,
  "results": [
    {"image_id": 1, "steps": {"preprocess": "ok", "roi": "ok", "feature": "ok", "result": "pending"}, "status": "ok"},
    {"image_id": 2, "steps": {"preprocess": "ok", "roi": "ok", "feature": "ok", "result": "pending"}, "status": "ok"}
  ]
}
```

### 示例 2：对一张新检测图执行浓度检测

```
POST /api/detect/3
```

响应：

```json
{
  "ok": true,
  "detection": {
    "id": 7, "image_id": 3,
    "conc": 42.1834, "u": 3.7219,
    "status": "within",
    "feature": "hue", "feature_value": 34.55,
    "model_name": "poly2-hue", "model_r2": 0.9921,
    "limits": {"lower": 0.0, "upper": 100.0}
  }
}
```

### 示例 3：首页总览与记录检索

```
GET /api/home/summary
```

```json
{
  "model_name": "poly2-hue", "model_type": "poly2",
  "model_r2": 0.9921, "model_lod": 1.27,
  "calibrated": true, "n_groups": 5, "n_points": 15,
  "last_detection": {"image_id": 3, "conc": 42.1834, "u": 3.7219, "status": "within", "model_name": "poly2-hue"},
  "today_count": 1, "total_detections": 1,
  "current_batch": null, "data_dir": "D:\\...\\fluro_app\\data"
}
```

```
GET /api/detections?batch=批次A
GET /api/detections/export      →  CSV：ID, 图片ID, 批次, 浓度C, 不确定度U, 判定, 检测时间
```

### 主要 API 一览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | /api/images/upload | 批量上传（multipart：files、kind、batch、known_conc） |
| POST | /api/pipeline/run | 批量执行自动流水线（可指定 image_ids） |
| POST | /api/pipeline/{id}/preprocess | 重跑预处理（可调滤波方式/核大小/暗场/平场） |
| POST | /api/pipeline/{id}/roi | 保存手动 ROI（级联重算特征与结果） |
| POST | /api/pipeline/{id}/roi/auto | 用激活模板自动套用 ROI |
| POST | /api/pipeline/{id}/features | 计算特征（平均 RGB、色相、饱和度、G/R、B/G、强度、纹理熵） |
| POST | /api/detect/{id} | 浓度检测：输出 C±U 与判定 |
| GET | /api/calibration/data | 标定分组数据与生效模型 |
| POST | /api/calibration/fit | 拟合 4 类模型并返回 R²/RMSE/LOD 与曲线点 |
| POST | /api/models | 保存生效模型（首个自动激活） |
| POST | /api/models/{id}/activate | 切换生效模型 |
| GET | /api/home/summary | 首页总览聚合 |
| GET | /api/detections、/api/detections/export | 记录检索、CSV 导出 |
| POST | /api/backup、/api/backup/restore | 一键备份、恢复 |

## 目录结构

```
D:\Doubao_APP\APP\
├── PRD.md                     # 产品需求文档（v1.1，含第一版完成定义）
├── DEVELOPMENT_PLAN.md        # M0~M9 开发计划
├── 开发计划-速览.html           # 可交互计划速览
├── 荧光浓度建模APP-页面结构与交互设计.html  # 可交互页面设计稿
└── fluro_app\                 # 应用本体
    ├── start.bat              # 一键启动（双击）
    ├── server.py              # 服务入口
    ├── requirements.txt       # 依赖
    ├── app\                   # 后端：配置、数据库、图像处理、建模、流水线、备份、路由
    ├── static\                # 前端：页面、样式、脚本（首页/工作流/建模/记录/设置）
    ├── tests\                 # 93 项测试（M0~M9）
    ├── tools\                 # 合成测试图生成工具
    ├── .venv\                 # 虚拟环境（不入版本库）
    ├── data\                  # 运行数据（不入版本库）
    └── backups\               # 备份（不入版本库）
```

## 已知限制（第一版范围外）

- 仅单人本机使用：无登录、联网、云数据库、多端同步、手机端。
- 一张图对应一条试纸，不支持一张图内多条试纸。
- 检测区定位采用「模板记忆 + 模板匹配」，不使用深度学习。
- 无报告自动排版与打印、无仪器硬件对接。
- 高浓度端（偏红色）色相测量噪声较大；若标定曲线在红端拟合不佳，可在建模页改用 G/R、B/G 通道比值等特征。
