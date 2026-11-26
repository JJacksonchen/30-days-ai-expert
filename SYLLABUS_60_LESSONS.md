# 《30天从0转行AI：从小白到资深AI专家》60期硬核体系总课表

> **课程目标**：30 天 × 每天 2 节（早 07:00 + 晚 17:00）= **60 期深度体系课**，每课 **25–30 分钟**，覆盖从零基础数学与编程直觉、深度学习底层原理、现代大模型与 Agent 系统架构、分布式训练与推理优化，到真实工业项目落地与资深 AI 工程师面试答辩。
> **教学准则**：**底层原理讲透（数学直觉 + 维度算账 + 真实代码 + 踩坑清单）× 小白零门槛听懂（生活类比 + 一镜到底动画可视化）**。

---

## 第一阶段（Day 1–5 · 第 01–10 课）：底层数学直觉、神经网络本质与深度学习基石
- **第 01 课（Day 1 上）**：从 0 看懂 AI 全局与数学底层——向量、矩阵、空间变换、张量维度 `[B, T, D]` 与神经网络本质（含 PyTorch 手写第一个神经元）
- **第 02 课（Day 1 下）**：机器学习核心引擎全解——损失函数选型（MSE/CE/KL/InfoNCE）、梯度下降、手算反向传播链式法则与优化器演进（SGD→AdamW→Muon）
- **第 03 课（Day 2 上）**：深度网络核心部件与视觉基石——激活函数（ReLU→GELU→SwiGLU）、LayerNorm/RMSNorm、残差连接（ResNet→mHC）、CNN 卷积与 GPU 并行屋顶线模型
- **第 04 课（Day 2 下）**：大语言模型心脏解剖——Tokenizer（BPE）、Embedding、RoPE 旋转位置编码、Self-Attention 矩阵推导、GQA/MLA 与 KV Cache 显存算账
- **第 05 课（Day 3 上）**：万亿大模型现代架构全解——DeepSeekMoE 稀疏路由、MTP 多步预测、1M 长上下文（CSA+HCA）、Chinchilla 缩放定律与分布式并行（DP/TP/PP/EP）
- **第 06 课（Day 3 下）**：AI 工程基本功——PyTorch 自动求导机制（Autograd）、GPU 显存分布拆解（参数/梯度/优化器/激活值）、混合精度训练（FP16/BF16/FP8）与梯度检查点
- **第 07 课（Day 4 上）**：数据处理与特征工程大全——从文本清洗、去重（MinHash）、数据配比到图像/音频增强，避免数据泄露与模型崩溃（Model Collapse）
- **第 08 课（Day 4 下）**：模型泛化与调参实战——过拟合/欠拟合诊断、学习率调度（Warmup + Cosine）、Batch Size 缩放律、正则化（Dropout/Weight Decay）与评价指标全解
- **第 09 课（Day 5 上）**：经典序列与生成模型演进——RNN/LSTM、自编码器 VAE 到扩散模型（Diffusion / Flow Matching）底层去噪原理
- **第 10 课（Day 5 下）**：第一阶段综合实战——从零手写一个可训练的 Mini-GPT（200 行 PyTorch 逐行精讲 + 显存/Loss 排错指南）

## 第二阶段（Day 6–10 · 第 11–20 课）：大模型后训练（Post-Training）、对齐与慢思考推理
- **第 11 课（Day 6 上）**：预训练（Pre-training）工业全景——万卡集群通信瓶颈、AllReduce/AllGather、Loss Spike 救火实战
- **第 12 课（Day 6 下）**：监督微调（SFT）与高效微调——全参微调 vs LoRA/QLoRA 低秩矩阵原理、指令数据构造黄金法则
- **第 13 课（Day 7 上）**：人类偏好对齐核心算法——RLHF（PPO）四模型架构、Reward Hacking 陷阱与 DPO/SimPO 数学化简
- **第 14 课（Day 7 下）**：强化学习与慢思考革命——GRPO 组内相对优势、Test-Time Compute 隐藏草稿纸、长思维链（CoT）涌现机制
- **第 15 课（Day 8 上）**：模型蒸馏（Distillation）全解——白盒 Logits 软标签 vs 黑盒思维链蒸馏、KL 散度温度系数 T 调优实战
- **第 16 课（Day 8 下）**：合成数据与铁面验证器——代码沙盒、单元测试与形式化验证（Lean 4）如何锁死大模型幻觉
- **第 17 课（Day 9 上）**：长上下文与记忆架构——位置插值 YaRN、Ring Attention、开源记忆救星 Engram 与外部记忆检索
- **第 18 课（Day 9 下）**：大模型评测体系与刷榜防坑——四大主流榜单拆解、数据污染检测、LLM-as-a-Judge 偏差校正与业务评测集搭建
- **第 19 课（Day 10 上）**：多模态大模型底层原理——CLIP 对比学习、ViT 视觉编码、端到端多模态融合与统一视觉架构
- **第 20 课（Day 10 下）**：第二阶段综合实战——用单卡 LoRA + GRPO 复现一个领域专用推理小模型

## 第三阶段（Day 11–16 · 第 21–32 课）：RAG、AI Agent 工程、Harness 架构与 MCP 生态
- **第 21 课（Day 11 上）**：现代 RAG 底层原理与向量检索——Embedding 空间、HNSW 图索引、BM25 混合检索与 Rerank 重排
- **第 22 课（Day 11 下）**：企业级 RAG 踩坑与进阶——PDF 复杂版面解析、文档切块策略、GraphRAG 知识图谱与召回率归因排查
- **第 23 课（Day 12 上）**：提示词工程到提示词编译——结构化输出、Few-Shot、DSPy 自动编译优化与 SkillOpt 技能自进化
- **第 24 课（Day 12 下）**：AI Agent 核心引擎——Function Calling 协议、ReAct 观察-规划-执行-反思闭环与状态机编排
- **第 25 课（Day 13 上）**：彻底讲透 Agent Harness——`Agent = Model + Harness`、一切皆插件架构、`historylist` 全链路追踪与 Token 账单控制
- **第 26 课（Day 13 下）**：AI 界的 USB-C——MCP 协议底层 JSON-RPC 通信、Server/Client 开发与权限安全隔离
- **第 27 课（Day 14 上）**：安全沙箱与 Computer Use——Docker/MicroVM 容器隔离、浏览器自动化与桌面 GUI 视觉操作原理
- **第 28 课（Day 14 下）**：多智能体（Multi-Agent）协同架构——主控拆解、并行工作流、交接协议、冲突仲裁与防止无限死循环
- **第 29 课（Day 15 上）**：Coding Agent 深度拆解——代码库 AST 索引、精准 Diff 编辑、自动跑测修复与 Vibe Coding 工程边界
- **第 30 课（Day 15 下）**：Agent 记忆系统设计——短期工作记忆、上下文压缩摘要、长期 episodic/semantic 记忆存储
- **第 31 课（Day 16 上）**：Agent 稳定性与可观测性——Trace 追踪、失败重试、成本护栏、高危操作 Human-in-the-loop 审批闸门
- **第 32 课（Day 16 下）**：第三阶段综合实战——从零搭建一个带沙箱、MCP 与全链路回放的自动化工程 Agent

## 第四阶段（Day 17–22 · 第 33–44 课）：高性能推理、量化部署与 AI 系统架构（MLOps）
- **第 33 课（Day 17 上）**：大模型推理性能瓶颈——Prefill（算力受限）vs Decode（显存带宽受限）、TTFT 与 TPOT 核心指标
- **第 34 课（Day 17 下）**：vLLM 与 PagedAttention 原理——像操作系统虚拟内存一样切分 KV Cache、Continuous Batching 吞吐暴涨真相
- **第 35 课（Day 18 上）**：FlashAttention 硬件级加速——GPU SRAM vs HBM 分级存储、分块平铺（Tiling）与在线 Softmax 原理
- **第 36 课（Day 18 下）**：模型量化（Quantization）彻底讲透——INT8/INT4/FP8、GPTQ、AWQ 激活感知权重保护与精度损失评估
- **第 37 课（Day 19 上）**：推理加速黑科技——投机采样（Speculative Decoding）、MTP 协同加速与前缀缓存（Prefix Caching）
- **第 38 课（Day 19 下）**：硬件选型与本地部署指南——按任务与预算挑 GPU/Apple Silicon、显存容量精确速算公式
- **第 39 课（Day 20 上）**：高并发大模型网关设计——多模型路由、限流熔断、语义缓存、流式 SSE 与成本监控
- **第 40 课（Day 20 下）**：AI 应用安全攻防——提示词注入（Prompt Injection）、越狱攻防、数据脱敏与合规护栏
- **第 41 课（Day 21 上）**：实时语音与视频交互架构——端到端多模态语音流（VoiceMem）、低延迟 VAD 与全双工打断
- **第 42 课（Day 21 下）**：端侧 AI 与边缘计算——手机/PC 端小模型压缩、NPU 加速与隐私优先本地智能体
- **第 43 课（Day 22 上）**：AI 产品数据飞轮——埋点日志回收、坏例自动聚类、在线 A/B 测试与持续微调闭环
- **第 44 课（Day 22 下）**：第四阶段综合实战——部署一套支持千人并发的量化 vLLM + 路由网关生产服务

## 第五阶段（Day 23–26 · 第 45–52 课）：前沿突破——世界模型、具身智能 VLA、科学 AI 与递归自学习
- **第 45 课（Day 23 上）**：世界模型（World Models）底层逻辑——从视频预测到物理因果模拟、JEPA 联合嵌入预测架构
- **第 46 课（Day 23 下）**：具身智能与 VLA（视觉-语言-动作）——动作 Token 化、Diffusion Policy 与全球人形机器人落地现状
- **第 47 课（Day 24 上）**：具身导航（VLN）与 Sim2Real——虚拟仿真训练如何跨过真实世界物理鸿沟
- **第 48 课（Day 24 下）**：AI for Science 革命——AlphaFold 蛋白质折叠原理、分子生成与自动化湿实验室闭环
- **第 49 课（Day 25 上）**：大模型可解释性（Mechanistic Interpretability）——稀疏自编码器 SAE 如何剖开神经网络黑盒找“概念神经元”
- **第 50 课（Day 25 下）**：递归自学习（RSI）前沿实证——AI 写自身源码（AIDE²）、AI 设计下一代芯片（AlphaChip）与评估器天花板
- **第 51 课（Day 26 上）**：2026 顶级论文精读方法论——如何 30 分钟拆透一篇 NeurIPS/ICLR/DeepSeek 顶会论文并复现核心模块
- **第 52 课（Day 26 下）**：AI 行业商业模式与技术护城河——算力层、模型层、Harness/应用层的价值分配真相

## 第六阶段（Day 27–30 · 第 53–60 课）：资深 AI 工程师求职转行、简历作品集与架构答辩
- **第 53 课（Day 27 上）**：转行 AI 四大黄金赛道定位——算法研发、AI 应用/Agent 工程、推理系统优化、数据/评测专家能力矩阵
- **第 54 课（Day 27 下）**：打造杀手级 GitHub 作品集——如何把 3 个端到端硬核项目写出“大厂 Senior 级”工程质感
- **第 55 课（Day 28 上）**：大厂 AI 面试高频手撕题（上）——Attention/MHA/GQA/RoPE/LoRA/Beam Search 纯 PyTorch 手撕精讲
- **第 56 课（Day 28 下）**：大厂 AI 面试高频八股与深度追问（下）——显存计算、梯度消失、RLHF 稳定性、RAG 幻觉排查标准答题框架
- **第 57 课（Day 29 上）**：AI 系统设计面试实战（System Design）——现场设计千万日活智能客服、企业知识库与自动化编程平台
- **第 58 课（Day 29 下）**：简历精修与面试反客为主——非科班/传统后端/产品转行如何把过往经验包装成稀缺 AI 优势
- **第 59 课（Day 30 上）**：资深 AI 专家的长期进化系统——如何跟踪前沿、避免被工具迭代淘汰、建立个人技术护城河
- **第 60 课（Day 30 下）**：结业大考与 30 天全景复盘——串起从向量矩阵到递归自学习的完整 AI 认知大厦！
