# LP-EM 图像处理深度学习代码写作流程（模型建立部分）

## 适用范围
- 仅覆盖训练阶段的模型建立主链。
- 不覆盖推理与后处理相关脚本：val_main.py、val_support.py、metrics.py。
- 目标是给后续复用提供一份可执行、可检查的写作流程。

## 总流程顺序
1. 配置契约定义
2. 数据契约与命名规则
3. Dataset 单样本拼装
4. 模型与损失注册
5. 训练装配 support 与 data_deal
6. 训练入口主流程串联
7. 训练循环闭环与产物落盘
8. 复用检查清单

---

## 步骤 1：配置契约定义
### 本步要做什么
- 在 config_file.py 中先定义全量参数，按模块分组：model、data、optimizer、scheduler、runtime。
- 明确每个参数的默认值、可选值、类型，以及后续消费位置。

### 本步输出
- 一个结构稳定的 parse_args。
- 可直接转换为字典并被后续 support/data_deal/train_process 消费的 config。

### 可考虑使用的函数
- parse_args（config_file.py）
- str2bool（utils.py）

### 完成检查
- 能列出 input_channels、num_classes、img_ext、mask_ext、arch、loss、scheduler 的定义位置。
- 参数名与后续代码消费名完全一致。

---

## 步骤 2：数据契约与命名规则
### 本步要做什么
- 先固定数据目录结构：inputs/{dataset}/images 与 inputs/{dataset}/masks/0。
- 固定样本命名规则：img_id 前缀 + 数字编号。
- 确认图片和标签后缀配置与真实数据一致。

### 本步输出
- 可被 train_main 采样并被 Dataset 正确解析的 img_id 列表。
- 邻域取帧逻辑可运行的命名规则。

### 可考虑使用的函数
- glob（train_main.py 中用于收集样本）
- train_test_split（train_main.py 中用于划分训练/验证）
- os.path.join
- os.path.splitext
- os.path.basename

### 完成检查
- 任意一个 img_id 都能被解析为前缀和数字部分。
- 对应图像与 mask 路径都可被拼接并访问。

---

## 步骤 3：Dataset 单样本拼装
### 本步要做什么
- 在 Dataset.__getitem__ 中完成：
  - 配置读取（input_channels）
  - 邻域图读取与缺失回退
  - mask 读取与缺失回退
  - image 与 mask 同步增强
  - 单通道抽取并堆叠
  - 数值归一化与 CHW 转换

### 本步输出
- 单样本输出格式：img、mask、meta。
- img 通道数与 input_channels 一致。

### 可考虑使用的函数
- Dataset.__len__（dataset.py）
- Dataset.__getitem__（dataset.py）
- yaml.load
- cv2.imread
- os.path.exists
- random.randint
- np.dstack
- ndarray.transpose

### 完成检查
- img 最终形状为 CxHxW。
- mask 最终形状为 1xHxW。
- 邻域缺失时能正确回退，不影响训练流程。

---

## 步骤 4：模型与损失注册
### 本步要做什么
- 在 archs.py 中维护模型注册表，确保可按配置名动态构建。
- 在 losses.py 中维护损失注册表，确保可按配置名动态构建。
- 确保模型输出形状与损失输入期望一致。

### 本步输出
- 模型可选集合与损失可选集合。
- 能由配置字段 arch/loss 直接索引并实例化。

### 可考虑使用的函数
- UNet、NestedUNet（archs.py）
- VGGBlock（archs.py）
- BCEDiceLoss、LovaszHingeLoss（losses.py）
- binary_cross_entropy_with_logits（losses.py 中调用）
- torch.sigmoid（losses.py 中调用）

### 完成检查
- arch 与 loss 的 choices 与注册集合一致。
- 深监督开关下模型输出与训练分支逻辑兼容。

---

## 步骤 5：训练装配 support 与 data_deal
### 本步要做什么
- 在 support 中完成训练前装配：
  - 目录准备
  - 配置落地
  - 损失函数实例化
  - 模型实例化
  - 优化器与学习率调度器创建
  - 日志结构初始化
- 在 data_deal 中完成数据增强、Dataset 构建、DataLoader 构建。

### 本步输出
- criterion、model、optimizer、scheduler、log。
- train_loader、val_loader。

### 可考虑使用的函数
- support（train_support.py）
- data_deal（train_support.py）
- Compose、OneOf（albumentations）
- A.Resize、A.RandomRotate90
- transforms.Flip、transforms.Normalize
- torch.utils.data.DataLoader
- optim.Adam、optim.SGD
- lr_scheduler.CosineAnnealingLR
- lr_scheduler.ReduceLROnPlateau
- lr_scheduler.MultiStepLR

### 完成检查
- support 先执行成功并写出 config/config.yml。
- DataLoader 可产出 batch，且 batch 形状与模型输入匹配。

---

## 步骤 6：训练入口主流程串联
### 本步要做什么
- 在 train_main 中固定调用顺序：
  1) 解析参数
  2) support 训练装配
  3) 收集并切分样本
  4) data_deal 获取 loader
  5) train_process 执行训练
  6) 统计并回写 runtime 与训练配置

### 本步输出
- 单次训练可完整跑通的主入口流程。

### 可考虑使用的函数
- computation_time（train_main.py）
- parse_args（config_file.py）
- support（train_support.py）
- data_deal（train_support.py）
- train_process（train_support.py）
- yaml.dump（train_main.py）

### 完成检查
- 调用顺序与依赖顺序一致，无前置缺失。
- 能在 models/{name} 下写出 model.pth、log.csv、config.yml。

---

## 步骤 7：训练循环闭环与产物落盘
### 本步要做什么
- 在 train/validate 中执行前向、损失、指标统计。
- 在 train_process 中处理：
  - 学习率调度步进
  - 日志累积并落盘
  - 最优模型保存
  - 早停触发

### 本步输出
- 可追踪训练过程的日志文件。
- 最优权重模型文件。

### 可考虑使用的函数
- train（train_support.py）
- validate（train_support.py）
- train_process（train_support.py）
- AverageMeter.update（utils.py）
- iou_score（train_support.py 中调用）
- torch.save
- torch.cuda.empty_cache
- pandas.DataFrame.to_csv

### 完成检查
- 每个 epoch 都有可解释的损失与指标输出。
- 最优模型仅在 val_iou 提升时更新。

---

## 步骤 8：复用检查清单（开新项目时直接照此检查）
### 必查一致性
- input_channels 是否同时匹配 Dataset 输出通道数和模型输入通道数。
- img_ext/mask_ext 是否与真实数据后缀一致。
- deep_supervision 开关是否与 train/validate 分支一致。
- scheduler 相关参数是否与对应调度器类型匹配。

### 必查时序
- support 必须先于第一次 DataLoader 取样执行。
- config/config.yml 必须可读且与当前运行参数一致。

### 当前代码可选优化点（非必须）
- Dataset 每次取样读取一次 config，可考虑改为初始化阶段缓存。
- mask 临时列表可简化为直接赋值，减少冗余中转。
- masks 目录类别当前写死为 0，如扩展多类需按类别循环组织。

---

## 一句话复用模板
先定配置契约，再定数据契约；先打通 Dataset 单样本，再注册模型与损失；然后用 support/data_deal 完成训练装配，最后在 train_main 严格按依赖顺序串联训练主流程。