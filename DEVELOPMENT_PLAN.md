# 荧光浓度建模 APP · 开发计划

> 依据：`PRD.md v1.1`（唯一需求依据）｜ 目标：完成并交付第一版，满足 PRD 第 10 节验收标准与第 11 节完成定义
> 本文档给出：技术方案、项目结构、数据设计、接口设计、算法方案、分阶段执行任务（每项含产物与验收）、测试策略、风险与交付物。

---

## 1. 技术方案总览

### 1.1 架构（方案 B：本地小服务）

```
┌──────────────────────────────────────────────┐
│ 浏览器（前端：单页 HTML/CSS/JS，无框架无 CDN） │
│  5 个页面 + Canvas 交互（ROI 框选/对比滑块）    │
└───────────────┬──────────────────────────────┘
                │ HTTP（仅访问 http://127.0.0.1:端口）
┌───────────────▼──────────────────────────────┐
│ 本地小服务（Python）                           │
│  API 层    ：Flask 路由（/api/*）             │
│  业务层    ：流水线编排 / 建模 / 记录 / 备份    │
│  图像处理  ：OpenCV + numpy                   │
│  建模统计  ：scipy + numpy                    │
│  数据存储  ：SQLite + 数据文件夹               │
└───────────────┬──────────────────────────────┘
                │ 读写
┌───────────────▼──────────────────────────────┐
│ data\ 数据目录（用户可见、可备份）              │
│  images\ thumbnails\ processed\ fluro.db     │
└──────────────────────────────────────────────┘
启动：start.bat 双击 → 启动服务 → 自动打开浏览器
```

### 1.2 技术选型（理由）

| 组件 | 选型 | 理由 |
| --- | --- | --- |
| 语言/运行时 | Python 3.10+ | 图像与科学计算生态最全；若本机无 Python，M0 阶段安装或由用户确认 |
| 后端框架 | Flask | 轻量、零额外依赖、适合单机小服务 |
| 图像处理 | opencv-python + numpy | 预处理、ROI、颜色特征的标准方案，性能满足验收基线 |
| 建模统计 | scipy（curve_fit）+ numpy | 4PL/指数等非线性拟合、协方差矩阵（不确定度传播） |
| 数据存储 | SQLite（标准库 sqlite3） | 单文件、零配置、事务安全，满足持久化要求 |
| 备份 | zipfile + shutil | 一键打包数据目录为单个 zip |
| 前端 | 原生 HTML/CSS/JS + Canvas | 离线可用（不依赖 CDN）、无构建步骤、直接可维护 |
| 启动 | start.bat + 浏览器自动打开 | 双击即用（webbrowser 模块） |

### 1.3 项目目录结构

```
荧光浓度建模APP\            # 项目根目录
├── PRD.md                        # 需求依据（已有）
├── DEVELOPMENT_PLAN.md           # 本文档
├── fluro_app\                    # 应用根目录
│   ├── start.bat                 # 双击启动（M0）
│   ├── requirements.txt          # 依赖清单（M0）
│   ├── README.md                 # 使用说明（M9）
│   ├── server.py                 # 服务入口：启动 Flask、自动开浏览器（M0/M1）
│   ├── app\
│   │   ├── __init__.py
│   │   ├── config.py             # 数据目录、端口、默认参数（M1）
│   │   ├── database.py           # SQLite 连接、建表、通用读写（M1）
│   │   ├── image_processing.py   # 预处理/ROI/特征提取（M2-M3）
│   │   ├── modeling.py           # 4 类拟合、R²/RMSE/LOD、不确定度（M5）
│   │   ├── pipeline.py           # 5 步流水线编排与级联重算（M4）
│   │   ├── backup.py             # 备份/恢复（M8）
│   │   └── routes\
│   │       ├── __init__.py       # 蓝图注册（M1）
│   │       ├── api_home.py       # 首页聚合（M7）
│   │       ├── api_workflow.py   # 上传/流水线/ROI/特征/结果（M1-M6）
│   │       ├── api_model.py      # 标定建模（M5）
│   │       ├── api_records.py    # 记录检索/详情/导出（M7）
│   │       └── api_settings.py   # 设置/模板/备份恢复（M8）
│   ├── static\
│   │   ├── index.html            # 单页骨架（M1）
│   │   ├── css\style.css
│   │   └── js\
│   │       ├── api.js            # fetch 封装
│   │       ├── app.js            # 导航/路由/全局状态条
│   │       ├── views\home.js     # 首页（M7）
│   │       ├── views\workflow.js # 工作流 5 步（M4）
│   │       ├── views\modeling.js # 标定建模（M5）
│   │       ├── views\records.js  # 检测记录（M7）
│   │       ├── views\settings.js # 设置（M8）
│   │       └── canvas\roi.js     # ROI 框选/拖动/缩放（M3）
│   └── data\                     # 运行时数据目录（M1 初始化，位置可配置）
│       ├── images\               # 原始图片（按日期子目录）
│       ├── thumbnails\
│       ├── processed\            # 处理后图与 ROI 叠加图
│       ├── fluro.db              # SQLite
│       └── backups\              # 备份文件输出
```

---

## 2. 数据设计（SQLite 表，M1 实现）

| 表名 | 关键字段 | 用途 |
| --- | --- | --- |
| images | id, file_path, thumb_path, kind(calibration/detection), batch, known_conc, status, created_at | 图片元数据与标记 |
| pipeline_steps | image_id, step(preprocess/roi/feature/result), params_json, status(ok/error/manual), updated_at | 每步参数快照（留痕） |
| features | image_id, mean_r, mean_g, mean_b, hue, saturation, value, ratio_gr, ratio_bg, intensity, texture_entropy | 颜色与纹理特征 |
| roi | image_id, x, y, w, h（归一化 0~1）, source(auto/manual), updated_at | 每张图检测区 |
| roi_templates | id, name, x, y, w, h（归一化）, ref_image_id, created_at | ROI 模板库 |
| calibration_groups | id, name, conc, created_at | 标定浓度分组 |
| calibration_points | id, group_id, image_id, included(1/0) | 分组内数据点（可剔除） |
| models | id, name, type(linear/poly2/exp/4pl), params_json, metrics_json(r2/rmse/lod), is_active, source_snapshot_json, created_at | 模型库与历史版本 |
| detections | id, image_id, model_id, conc, u, status(normal/over/under), params_snapshot_json, batch, created_at | 检测结果（含快照） |
| settings | key, value | 判定上下限、置信水平、单位、数据目录等 |
| batches | id, name, created_at | 批次登记 |

**关键约定**
- ROI 一律存**归一化坐标**（0~1），与图片分辨率解耦，换图可复用。
- 检测结果快照 `params_snapshot_json` 保存：预处理参数、ROI 坐标、特征值、模型 id/参数、判定限——满足"可追溯"。

---

## 3. API 设计概要（前端 ↔ 本地服务）

| 方法 | 路径 | 功能 | 归属 |
| --- | --- | --- | --- |
| POST | /api/images/upload | 批量上传（multipart），携带 kind/batch/conc | M1 |
| GET | /api/images/{id}/original \| /processed \| /roi-overlay | 取图（原图/处理后/ROI 叠加） | M2/M3 |
| POST | /api/pipeline/{id}/run | 触发该图自动流水线（预处理→ROI→特征→结果） | M4 |
| POST | /api/pipeline/{id}/preprocess | 改参数重跑预处理（级联） | M2 |
| GET/POST | /api/pipeline/{id}/roi | 获取/保存 ROI（手动修正） | M3 |
| GET | /api/pipeline/{id}/features | 特征值 | M3 |
| POST | /api/pipeline/{id}/result | 重新预测（换模型/改限值） | M6 |
| GET/POST | /api/calibration/groups | 浓度分组管理 | M5 |
| GET/POST | /api/calibration/points | 数据点纳入/剔除 | M5 |
| POST | /api/calibration/fit | 拟合 4 类模型，返回 R²/RMSE/LOD | M5 |
| POST | /api/models/{id}/activate | 设为生效模型 | M5 |
| GET | /api/records?time=&conc_min=&conc_max=&batch=&status= | 组合筛选 | M7 |
| GET | /api/records/{id} | 记录详情（含快照） | M7 |
| GET | /api/records/export?… | 导出 CSV | M7 |
| GET | /api/home/summary | 首页聚合数据 | M7 |
| GET/POST | /api/settings | 读取/保存设置 | M8 |
| GET/POST | /api/templates | ROI 模板管理 | M8 |
| POST | /api/backup | 一键备份（返回 zip） | M8 |
| POST | /api/restore | 从备份恢复 | M8 |

---

## 4. 关键算法方案

### 4.1 预处理（M2）
- **暗场校正**：`I' = (I − dark) / (flat − dark)`；dark 为暗场参考图，flat 为平场参考图；二者可选，缺省时仅做去噪。
- **去噪**：高斯或中值滤波，`kernel`（3/5/7）与 σ 可调；提供"重跑本步"接口，返回处理后图。
- **对比**：前端 Canvas 滑块左右对照原图/处理后图。

### 4.2 ROI 分割（M3）
- **模板记忆**：参考图（首图）框选 → 存归一化坐标模板；新图自动套用。
- **自动对齐**：固定机位下直接映射；再用 `cv2.matchTemplate`（以模板区域为参考）做 ±小范围微调，适配轻微位移。
- **手动修正**：Canvas 拖拽/缩放选区，保存为该图 ROI（source=manual）。

### 4.3 特征提取（M3，颜色为主）
- 检测区内：RGB 均值（mean_r/g/b）；转 HSV 得平均色相/饱和度/明度；
- 通道比值：G/R、B/G；荧光强度 = 明度均值；纹理熵 = 灰度直方图熵。
- 全部以检测区像素为计算范围，随 ROI 修正自动重算。

### 4.4 建模（M5）
- 拟合方向：`特征 = f(浓度)`；预测时反解浓度（4PL 有解析反函数；其余用数值求根/一维插值反演）。
- 4 类模型：线性 `y=a·x+b`；二次多项式；指数 `y=a·e^(b·x)+c`；4PL `y=d+(a−d)/(1+(x/c)^b)`。
- 非线性拟合用 `scipy.optimize.curve_fit`，多组初值尝试防不收敛。
- 指标：R²、RMSE（对特征域）；**LOD = 3.3·σ/斜率**（σ 取低浓度端重复测定的标准差，斜率取 4PL 线性段或线性拟合斜率；实现时在建模页注明口径）。
- 数据点支持"剔除离群点"后重拟合（calibration_points.included 标记）。

### 4.5 不确定度 U（M6）
- 用拟合协方差矩阵 + 残差方差计算 **95% 预测区间**，`U ≈ t(0.975, n−k)·SE_pred`（k=2 近似），对特征域求预测带后反演到浓度域。
- 结果输出 `C ± U`（默认 95%，k=2，设置页可改置信水平）。

### 4.6 判定（M6）
- `settings` 存判定上下限；`conc > upper` → over，`conc < lower` → under，否则 normal；超限醒目提示。
- 换模型/改限值后重新预测，结果与快照同步更新。

---

## 5. 分阶段执行计划（可直接执行）

> 每阶段结束必须跑通该阶段验收项（映射 PRD 第 10 节编号），才能进入下一阶段。
> 工作量标记：S 小（≤1 天）、M 中（1~2 天）、L 大（2~4 天）——按单人全栈计。

### M0 环境与骨架（S）
**任务**
1. 检查本机 Python 版本（`python --version`）；缺失则安装 Python 3.10+（勾选 Add to PATH），或与用户确认替代方案。
2. 创建 `fluro_app\` 目录结构与 `requirements.txt`（flask, opencv-python, numpy, scipy）。
3. 安装依赖并验证 `import cv2, numpy, scipy, flask` 成功。
4. 编写 `start.bat`：启动 `server.py` 并用 `webbrowser` 自动打开 `http://127.0.0.1:8000`；端口被占用时自动换端口。
5. `server.py` 返回最小页面（"服务已启动"）。
**验收**：双击 `start.bat` → 浏览器自动打开并显示页面；再次双击不冲突（端口复用/提示）。

### M1 数据层与基础页面（M）
**任务**
1. `config.py`：数据目录（默认 `fluro_app\data`，可配置）、端口、默认参数（滤波 kernel、置信水平 95%、判定限占位）。
2. `database.py`：建表（第 2 节全部表）、通用 CRUD、迁移策略（表结构变更时备份旧库）。
3. 图片入库：批量上传接口 → 保存原图到 `data\images\YYYYMMDD\`、生成缩略图（OpenCV resize）、写入 images 表。
4. `index.html` + `app.js`：左侧 5 导航、顶部全局状态条、页面容器；`api.js` fetch 封装。
5. 测试用合成图生成器（开发/验收辅助）：按已知浓度生成"颜色随浓度变化"的模拟荧光试纸图（例如色相随浓度线性偏移的圆形检测区 + 背景），存为 `tools\make_test_images.py`——用于用户提供真实图之前验证全流水线。
**验收**：批量上传 ≥10 张测试图，缩略图墙正确显示（PRD B1）；导航可切换 5 个占位页面。

### M2 预处理模块（M）
**任务**
1. `image_processing.py`：暗场校正、平场校正、高斯/中值去噪函数。
2. `api_workflow.py`：`/preprocess` 接口（参数：use_dark/use_flat/kernel/滤波方式），返回处理后图 URL。
3. 工作流页"预处理"步骤 UI：参数控件 + 前后对比滑块 + "重跑本步"。
4. 预处理参数写入 pipeline_steps（快照）。
**验收**：对测试图调节 kernel/滤波方式并重跑，输出图明显变化；前后对比滑块可对照（PRD C1 前半、C4）。

### M3 ROI 与特征提取（M）
**任务**
1. 后端：ROI 模板 CRUD；自动套用（映射 + matchTemplate 微调）；ROI 存取（归一化坐标）。
2. 前端 `canvas\roi.js`：在图上框选、拖动、缩放检测区；保存修正。
3. `/features` 接口与特征表 UI（平均 RGB/色相/饱和度/通道比值/强度/纹理熵）。
4. ROI 修正 → 特征重算（级联）。
**验收**：首次框选存模板；第二张测试图自动套用；手动修正后特征更新（PRD C2、C3、D1）。

### M4 自动流水线编排（L）
**任务**
1. `pipeline.py`：5 步编排（预处理 → ROI → 特征 → 结果），支持单图与批量、进度状态、失败图标记（ROI 失配等 → status=attention）。
2. 上传后自动触发流水线（前端自动调用 /run）。
3. 工作流页完整 UI：步骤条（高亮/完成/异常）、每步展开面板、异常图列表、级联重算（改上游 → 自动重跑下游）。
4. 结果步骤：检测模式输出 C±U（先接占位模型，M6 实装）；标定模式提示"加入标定数据集"。
**验收**：上传一批测试图 → 自动依次完成 5 步，无需手动操作（PRD B3）；改预处理参数后下游特征自动更新（C1 级联部分）。

### M5 标定建模（L）
**任务**
1. `modeling.py`：4 类模型拟合、R²/RMSE/LOD、多初值策略、预测反解。
2. 标定数据管理：浓度分组（增删改）、数据点纳入/剔除、每组均值±SD。
3. 建模页三栏 UI：数据表 / 模型对比表（推荐标记）/ 曲线图（散点+拟合+95% 预测带，用 Canvas 或轻量 SVG 自绘，不引外部图表库）。
4. 模型库：保存/激活/删除/历史版本；`/models/{id}/activate`。
5. 标定流程端到端：标定图 → 自动流水线 → 分组 → 拟合 → 保存生效模型。
**验收**：用合成测试图（≥5 个浓度、每组 2~3 张重复）完成一次完整标定；4 类模型 R²/RMSE/LOD 正确显示且曲线一致（PRD D2~D5）。

### M6 检测与判定（M）
**任务**
1. 预测管线：新图 → 特征 → 生效模型反解浓度 → 预测区间算 U（95%，k=2）。
2. 超限判定（上下限来自 settings）；结果页输出 C±U 与状态徽标。
3. 换模型/改限值 → 重新预测并更新快照（E3）。
4. 单张与批量检测结果列表。
**验收**：新测试图输出 C±U；人为设置上下限后超限/正常判定正确（PRD E1、E2、E3）。

### M7 首页与检测记录（M）
**任务**
1. `/api/home/summary`：最近 C±U、生效模型 R²/LOD、累计记录数、最近 10 条、异常提醒。
2. 首页 UI：4 信息卡、曲线缩略图（复用建模页绘图）、记录表、快捷操作、异常入口。
3. 记录页：组合筛选（时间/浓度/批次/状态）、详情抽屉（原图+ROI 叠加+特征+快照+模型）、CSV 导出。
4. 模块联动：信息卡跳转。
**验收**：首页数据与工作流/建模/记录实际一致（PRD G1、G2）；筛选命中正确、详情可回看、CSV 一致（F1~F3）。

### M8 设置与备份恢复（M）
**任务**
1. 设置页：判定限、置信水平、单位、数据目录显示；模型管理、ROI 模板管理、参考图管理、批次管理。
2. `backup.py`：一键备份（zip 打包 data 目录）→ 下载到用户指定位置；恢复（上传备份文件 → 解包校验 → 替换/合并数据）。
3. 备份文件含版本号与校验（zip 内 manifest.json）。
**验收**：备份→清空→恢复往返一致（PRD A2、A3）；重启后数据完整（A1）。

### M9 打磨、性能与交付验收（L）
**任务**
1. 性能：单张全流程 ≤5 秒、10 张批量 ≤60 秒（优先用真实图验证；不足则优化 resize/缓存/并行）。
2. 断网验证（A4）；P0/P1 缺陷清零；异常与边界处理（坏图、重复上传、空数据、端口占用）。
3. 按 PRD 第 10 节 A~G 逐项实测并记录结果表。
4. `README.md`：启动方式、数据目录位置、备份恢复、已知限制。
5. 交付物清单核对（PRD 第 11 节 10 条）。
**验收**：完成定义 10 条全部满足。

---

## 6. 依赖关系与建议顺序

```
M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9
       └────────┘ (M2/M3 可并行)
                 M4 依赖 M2+M3；M5 依赖 M3；M6 依赖 M5；M7 依赖 M4~M6
```
- M1 前不并行：建表与上传是后续一切的基础。
- M4（流水线编排）是集成点，尽量在 M2/M3 稳定后开始。
- M9 必须使用真实试纸图做最终验收；开发期用合成图，用户可在 M6 前后提供真实图替换验证。

---

## 7. 测试策略

| 层级 | 方式 | 覆盖 |
| --- | --- | --- |
| 单元 | pytest：图像处理、特征计算、4 类拟合（合成数据已知答案）、LOD/不确定度计算 | 算法正确性 |
| 接口 | curl + 浏览器实测每个 /api 接口 | 前后端契约 |
| 端到端 | 合成图跑通 标定+检测 两条流程 | 流水线 |
| 验收 | 按 PRD 第 10 节 A~G 逐项操作，记录结果与日期 | 完成定义 |
| 回归 | 每次改动后重跑 A~G 关键项（A1/A3/B3/C1/E1/F1/G1） | 防回归 |

---

## 8. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 本机无 Python/依赖安装失败 | 无法启动 | M0 首步检查；给出安装指引；必要时确认是否打包 exe（列入二期） |
| 4PL 等非线性拟合不收敛 | 建模失败 | 多组初值 + 参数边界 + 明确报错提示 |
| 颜色特征对曝光/背景敏感 | 结果不稳 | 暗场/平场校正 + ROI 限定检测区 + 参考图管理 |
| 真实图与合成图差异大 | 验收失真 | M9 强制用真实图验收；M6 前后向用户索取样例图 |
| 端口占用 / 重复启动 | 打不开页面 | 自动换端口 + 已运行提示 |
| 数据目录被误删/损坏 | 数据丢失 | 备份恢复 + manifest 校验 + README 强调备份 |

---

## 9. 交付物清单（M9 核对）

1. `fluro_app\` 完整应用（start.bat / server.py / app\ / static\）
2. `requirements.txt`
3. `README.md`（启动、数据目录、备份恢复、已知限制）
4. `tools\make_test_images.py`（合成测试图生成器，留档可复用）
5. `data\` 初始结构与示例数据（可选）
6. 验收记录（A~G 实测结果表，随 M9 产出）
7. 已更新的 `PRD.md v1.1`（如验收中发现需澄清处，回写 PRD 并升版）
