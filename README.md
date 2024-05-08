[**🇨🇳中文**](./README.md) | [**🌐English**](./README_EN.md) | [**📖文档/Docs**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki) | [**❓提问/Issues**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/issues) | [**💬讨论/Discussions**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/discussions) | [**⚔️竞技场/Arena**](http://llm-arena.ymcui.com/)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="800"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca-3.svg?color=blue&style=flat-square">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/ymcui/Chinese-LLaMA-Alpaca-3">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca-3">
    <a href="https://app.codacy.com/gh/ymcui/Chinese-LLaMA-Alpaca-3/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/142d688425494644b5b156068f55370d"/></a>
</p>

本项目基于Meta最新发布的新一代开源大模型[Llama-3](https://github.com/facebookresearch/llama3)开发，是Chinese-LLaMA-Alpaca开源大模型相关系列项目（[一期](https://github.com/ymcui/Chinese-LLaMA-Alpaca)、[二期](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)）的第三期。本项目开源了**中文Llama-3基座模型和中文Llama-3-Instruct指令精调大模型**。这些模型在原版Llama-3的基础上使用了大规模中文数据进行增量预训练，并且使用精选指令数据进行精调，进一步提升了中文基础语义和指令理解能力，相比二代相关模型获得了显著性能提升。

#### 主要内容

- 🚀 开源Llama-3-Chinese基座模型和Llama-3-Chinese-Instruct指令模型
- 🚀 开源了预训练脚本、指令精调脚本，用户可根据需要进一步训练或微调模型
- 🚀 开源了alpaca_zh_51k, stem_zh_instruction, ruozhiba_gpt4_turbo指令精调数据
- 🚀 提供了利用个人电脑CPU/GPU快速在本地进行大模型量化和部署的教程
- 🚀 支持[🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [vLLM](https://github.com/vllm-project/vllm), [Ollama](https://ollama.com)等Llama-3生态

----

[中文Mixtral大模型](https://github.com/ymcui/Chinese-Mixtral) | [中文LLaMA-2&Alpaca-2大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) | [中文LLaMA&Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | [多模态中文LLaMA&Alpaca大模型](https://github.com/airaria/Visual-Chinese-LLaMA-Alpaca) | [多模态VLE](https://github.com/iflytek/VLE) | [中文MiniRBT](https://github.com/iflytek/MiniRBT) | [中文LERT](https://github.com/ymcui/LERT) | [中英文PERT](https://github.com/ymcui/PERT) | [中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner) | [蒸馏裁剪一体化GRAIN](https://github.com/airaria/GRAIN)


## 新闻

**[2024/05/08]  发布Llama-3-Chinese-8B-Instruct-v2版指令模型，直接采用500万条指令数据在 [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 上进行精调。详情查看：[📚v2.0版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/releases/tag/v2.0)** 

[2024/05/07]  添加预训练脚本、指令精调脚本。详情查看：[📚v1.1版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/releases/tag/v1.1)

[2024/04/30]  发布Llama-3-Chinese-8B基座模型和Llama-3-Chinese-8B-Instruct指令模型。详情查看：[📚v1.0版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/releases/tag/v1.0)

[2024/04/19] 🚀 正式启动Chinese-LLaMA-Alpaca-3项目


## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| [💁🏻‍♂️模型简介](#模型简介) | 简要介绍本项目相关模型的技术特点 |
| [⏬模型下载](#模型下载)        | 中文Llama-3大模型下载地址 |
| [💻推理与部署](#推理与部署) | 介绍了如何对模型进行量化并使用个人电脑部署并体验大模型 |
| [💯模型效果](#模型效果) | 介绍了模型在部分任务上的效果    |
| [📝训练与精调](#训练与精调) | 介绍了如何训练和精调中文Llama-3大模型 |
| [❓常见问题](#常见问题) | 一些常见问题的回复 |


## 模型简介

本项目推出了基于Meta Llama-3的中文开源大模型Llama-3-Chinese以及Llama-3-Chinese-Instruct。主要特点如下：

#### 📖 使用原版Llama-3词表

- Llama-3相比其前两代显著扩充了词表大小，由32K扩充至128K，并且改为BPE词表
- 初步实验发现Llama-3词表的编码效率与我们扩充词表的[中文LLaMA-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)相当，效率约为中文LLaMA-2词表的95%（基于维基百科数据上的编码效率测试）
- 结合我们在[中文Mixtral](https://github.com/ymcui/Chinese-Mixtral)上的相关经验及实验结论[^1]，我们**并未对词表进行额外扩充**

[^1]: [Cui and Yao, 2024. Rethinking LLM Language Adaptation: A Case Study on Chinese Mixtral](https://arxiv.org/abs/2403.01851)

#### 🚄 长上下文长度由二代4K扩展至8K

- Llama-3将原生上下文窗口长度从4K提升至8K，能够进一步处理更长的上下文信息
- 用户也可通过PI、NTK、YaRN等方法对模型进行长上下文的扩展，以支持更长文本的处理

#### ⚡ 使用分组查询注意力机制

- Llama-3采用了Llama-2中大参数量版本应用的分组查询注意力（GQA）机制，能够进一步提升模型的效率

#### 🗒 全新的指令模板

- Llama-3-Instruct采用了全新的指令模板，与Llama-2-chat不兼容，使用时应遵循官方指令模板（见[指令模板](#指令模板)）

## 模型下载

### 模型选择指引

以下是本项目的模型对比以及建议使用场景。**如需聊天交互，请选择Instruct版。**

| 对比项                | Llama-3-Chinese-8B             | Llama-3-Chinese-8B-Instruct |
| :-------------------- | :----------------------------------------------------: | :----------------------------------------------------------: |
| 模型类型 | 基座模型 | 指令/Chat模型（类ChatGPT） |
| 模型大小 | 8B | 8B |
| 训练类型     | Causal-LM (CLM)           | 指令精调                                                     |
| 训练方式 | LoRA + 全量emb/lm-head | LoRA + 全量emb/lm-head |
| 初始化模型 | [原版Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | v1: Llama-3-Chinese-8B<br/>v2: [原版Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| 训练语料 | 无标注通用语料（约120GB） | 有标注指令数据（约500万条） |
| 词表大小 | 原版词表（128,256） | 原版词表（128,256） |
| 支持上下文长度 | 8K | 8K |
| 输入模板              | 不需要                                                 | 需要套用Llama-3-Instruct模板 |
| 适用场景            | 文本续写：给定上文，让模型生成下文            | 指令理解：问答、写作、聊天、交互等 |


### 下载地址

| 模型名称                  |                    完整版                    |                    LoRA版                    |                    GGUF版                    |
| :------------------------ | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **Llama-3-Chinese-8B-Instruct-v2**<br/>(指令模型) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2-lora)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2-lora)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2-lora) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2-gguf)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2-gguf) |
| **Llama-3-Chinese-8B-Instruct**<br/>(指令模型) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-lora)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-lora)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-lora) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-gguf)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-gguf) |
| **Llama-3-Chinese-8B**<br/>(基座模型) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-lora)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-lora)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-lora) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-gguf)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-gguf) |

模型类型说明：

- **完整模型**：可直接用于训练和推理，无需其他合并步骤
- **LoRA模型**：需要与基模型合并并才能转为完整版模型，合并方法：[**💻 模型合并步骤**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/model_conversion_zh)
  - v1基模型：原版[Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
  - v2基模型：原版[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- **GGUF模型**：[llama.cpp](https://github.com/ggerganov/llama.cpp)推出的量化格式，适配ollama等常见推理工具，推荐只需要做推理部署的用户下载；模型名后缀为`-im`表示使用了importance matrix进行量化，通常具有更低的PPL，建议使用（用法与常规版相同）
> [!NOTE]
> 若无法访问HF，可考虑一些镜像站点（如[hf-mirror.com](hf-mirror.com)），具体方法请自行查找解决。

## 推理与部署

本项目中的相关模型主要支持以下量化、推理和部署方式，具体内容请参考对应教程。

| 工具   | 特点     | CPU  | GPU  | 量化 | GUI  | API  | vLLM |                      教程                             |
| :----------------------------------------------------------- | ---------------------------- | :--: | :--: | :--: | :--: | :--: | :--: |:--: |
| [llama.cpp](https://github.com/ggerganov/llama.cpp)      | 丰富的GGUF量化选项和高效本地推理 |  ✅   |  ✅   |  ✅   |  ✅  |  ✅   |  ❌   | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/llamacpp_zh) |
| [🤗transformers](https://github.com/huggingface/transformers) | 原生transformers推理接口     |  ✅   |  ✅   |  ✅   |  ✅   |  ❌   |  ✅  | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/inference_with_transformers_zh) |
| [仿OpenAI API调用](https://platform.openai.com/docs/api-reference) | 仿OpenAI API接口的服务器Demo |  ✅   |  ✅   |  ✅   |  ❌   |  ✅   |  ✅  | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/openai_api_zh) |
| [text-generation-webui](https://github.com/oobabooga/text-generation-webui) | 前端Web UI界面的部署方式 |  ✅   |  ✅   |  ✅   |  ✅   |  ✅  | ❌  | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/text-generation-webui_zh) |
| [LM Studio](https://lmstudio.ai) | 多平台聊天软件（带界面） | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/lmstudio_zh) |
| [Ollama](https://github.com/ollama/ollama) | 本地运行大模型推理 | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/ollama_zh) |


## 模型效果

为了评测相关模型的效果，本项目分别进行了生成效果评测和客观效果评测（NLU类），从不同角度对大模型进行评估。推荐用户在自己关注的任务上进行测试，选择适配相关任务的模型。

### 生成效果评测

- 本项目仿照[Fastchat Chatbot Arena](https://chat.lmsys.org/?arena)推出了模型在线对战平台，可浏览和评测模型回复质量。对战平台提供了胜率、Elo评分等评测指标，并且可以查看两两模型的对战胜率等结果。**⚔️ 模型竞技场：[http://llm-arena.ymcui.com](http://llm-arena.ymcui.com/)**
- examples目录中提供了Llama-3-Chinese-8B-Instruct和Chinese-Mixtral-Instruct的输出样例，并通过GPT-4-turbo进行了打分对比，**Llama-3-Chinese-8B-Instruct平均得分为8.1、Chinese-Mixtral-Instruct平均得分为7.8**。**📄 输出样例对比：[examples](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/blob/main/examples)**
- 本项目已入驻机器之心SOTA!模型平台，后期将实现在线体验：https://sota.jiqizhixin.com/project/chinese-llama-alpaca-3


### 客观效果评测

#### C-Eval

[C-Eval](https://cevalbenchmark.com)是一个全面的中文基础模型评估套件，其中验证集和测试集分别包含1.3K和12.3K个选择题，涵盖52个学科。C-Eval推理代码请参考本项目：[📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/ceval_zh)

| Models             | Valid (0-shot) | Valid (5-shot) | Test (0-shot) | Test (5-shot) |
| ------------------------ | :-----------: | :-----------: | :-----------: | :-----------: |
| **Llama-3-Chinese-8B-Instruct-v2** | 51.6 | 51.6 | 49.7 | 49.8 |
| **Llama-3-Chinese-8B-Instruct** | 49.3 | 51.5 | 48.3 | 49.4 |
| **Llama-3-Chinese-8B** | 47.0 | 50.5 | 46.1 | 49.0 |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 51.3 | 51.3 | 49.5 | 51.0 |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | 49.3 | 51.2 | 46.1 | 49.4 |
| [Chinese-Mixtral-Instruct](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 51.7 | 55.0 | 50.0 | 51.5 |
| [Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 45.8 | 54.2 | 43.1 | 49.1 |
| [Chinese-Alpaca-2-13B](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) | 44.3 | 45.9 | 42.6 | 44.0 |
| [Chinese-LLaMA-2-13B](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) | 40.6 | 42.7 | 38.0 | 41.6 |

#### CMMLU

[CMMLU](https://github.com/haonan-li/CMMLU)是另一个综合性中文评测数据集，专门用于评估语言模型在中文语境下的知识和推理能力，涵盖了从基础学科到高级专业水平的67个主题，共计11.5K个选择题。CMMLU推理代码请参考本项目：[📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/cmmlu_zh)

| Models             | Test (0-shot) | Test (5-shot) |
| ------------------------ | :-----------: | :-----------: |
| **Llama-3-Chinese-8B-Instruct-v2** | 51.8 | 52.4 |
| **Llama-3-Chinese-8B-Instruct** | 49.7 | 51.5 |
| **Llama-3-Chinese-8B** | 48.0 | 50.9 |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 53.0 | 53.5 |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | 47.8 | 50.8 |
| [Chinese-Mixtral-Instruct](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 50.0 | 53.0 |
| [Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 42.5 | 51.0 |
| [Chinese-Alpaca-2-13B](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) |     43.2      |     45.5      |
| [Chinese-LLaMA-2-13B](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) |     38.9      |     42.5      |

#### MMLU

[MMLU](https://github.com/hendrycks/test)是一个用于评测自然语言理解能力的英文评测数据集，是当今用于评测大模型能力的主要数据集之一，其中验证集和测试集分别包含1.5K和14.1K个选择题，涵盖57个学科。MMLU推理代码请参考本项目：[📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/mmlu_zh)

| Models             | Valid (0-shot) | Valid (5-shot) | Test (0-shot) | Test (5-shot) |
| ------------------------ | :-----------: | :-----------: | :-----------: | :-----------: |
| **Llama-3-Chinese-8B-Instruct-v2** | 62.1 | 63.9 | 62.6 | 63.7 |
| **Llama-3-Chinese-8B-Instruct** | 60.1 | 61.3 | 59.8 | 61.8 |
| **Llama-3-Chinese-8B** | 55.5 | 58.5 | 57.3 | 61.1 |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 63.4 | 64.8 | 65.1 | 66.4 |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | 58.6 | 62.5 | 60.5 | 65.0 |
| [Chinese-Mixtral-Instruct](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 65.1 | 69.6 | 67.5 | 69.8 |
| [Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 63.2 | 67.1 | 65.5 | 68.3 |
| [Chinese-Alpaca-2-13B](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) | 49.6 | 53.2 | 50.9 | 53.5 |
| [Chinese-LLaMA-2-13B](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) | 46.8 | 50.0 | 46.6 | 51.8 |

#### LongBench

[LongBench](https://github.com/THUDM/LongBench)是一个大模型长文本理解能力的评测基准，由6大类、20个不同的任务组成，多数任务的平均长度在5K-15K之间，共包含约4.75K条测试数据。以下是本项目模型在该中文任务（含代码任务）上的评测效果。LongBench推理代码请参考本项目：[📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/longbench_zh)

| Models                                                       | 单文档QA | 多文档QA | 摘要 | FS学习 | 代码 | 合成 | 平均 |
| ------------------------------------------------------------ | :------: | :------: | :--: | :----: | :--: | :--: | :--: |
| **Llama-3-Chinese-8B-Instruct-v2**                           |   57.3   |   27.1   | 13.9 |  30.3  | 60.6 | 89.5 | 46.4 |
| **Llama-3-Chinese-8B-Instruct**                              |   44.1   |   24.0   | 12.4 |  33.5  | 51.8 | 11.5 | 29.6 |
| **Llama-3-Chinese-8B**                                       |   16.4   |   19.3   | 4.3  |  28.7  | 14.3 | 4.6  | 14.6 |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |   55.1   |   15.1   | 0.1  |  24.0  | 51.3 | 94.5 | 40.0 |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |   21.2   |   22.9   | 2.7  |  35.8  | 65.9 | 40.8 | 31.6 |
| [Chinese-Mixtral-Instruct](https://github.com/ymcui/Chinese-Mixtral) (8x7B) |   50.3   |   34.2   | 16.4 |  42.0  | 56.1 | 89.5 | 48.1 |
| [Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral) (8x7B) |   32.0   |   23.7   | 0.4  |  42.5  | 27.4 | 14.0 | 23.3 |
| [Chinese-Alpaca-2-13B-16K](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) |   47.9   |   26.7   | 13.0 |  22.3  | 46.6 | 21.5 | 29.7 |
| [Chinese-LLaMA-2-13B-16K](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) |   36.7   |   17.7   | 3.1  |  29.8  | 13.8 | 3.0  | 17.3 |
| [Chinese-Alpaca-2-7B-64K](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) |   44.7   |   28.1   | 14.4 |  39.0  | 44.6 | 5.0  | 29.3 |
| [Chinese-LLaMA-2-7B-64K](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) |   27.2   |   16.4   | 6.5  |  33.0  | 7.8  | 5.0  | 16.0 |

### Open LLM Leaderboard

[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)是由HuggingFaceH4团队发起的大模型综合能力评测基准（英文），包含ARC、HellaSwag、MMLU、TruthfulQA、Winograde、GSM8K等6个单项测试。以下是本项目模型在该榜单上的评测效果。

| Models                                                       |  ARC  | HellaS | MMLU  |  TQA  | WinoG | GSM8K | 平均  |
| ------------------------------------------------------------ | :---: | :----: | :---: | :---: | :---: | :---: | :---: |
| **Llama-3-Chinese-8B-Instruct-v2**                           | 62.63 | 79.72  | 66.48 | 53.93 | 76.72 | 60.58 | 66.68 |
| **Llama-3-Chinese-8B-Instruct**                              | 61.26 | 80.24  | 63.10 | 55.15 | 75.06 | 44.43 | 63.21 |
| **Llama-3-Chinese-8B**                                       | 55.88 | 79.53  | 63.70 | 41.14 | 77.03 | 37.98 | 59.21 |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 60.75 | 78.55  | 67.07 | 51.65 | 74.51 | 68.69 | 66.87 |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | 59.47 | 82.09  | 66.69 | 43.90 | 77.35 | 45.79 | 62.55 |
| [Chinese-Mixtral-Instruct](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 67.75 | 85.67  | 71.53 | 57.46 | 83.11 | 55.65 | 70.19 |
| [Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 67.58 | 85.34  | 70.38 | 46.86 | 82.00 | 0.00  | 58.69 |

*注：MMLU结果与不同的主要原因是评测脚本不同导致。*

### 量化效果评测

在llama.cpp下，测试了Llama-3-Chinese-8B（基座模型）的量化性能，如下表所示。实测速度相比二代Llama-2-7B略慢。

|               |   F16 |  Q8_0 |  Q6_K |  Q5_K |  Q5_0 |  Q4_K |  Q4_0 |  Q3_K |   Q2_K |
| ------------- | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | -----: |
| **Size (GB)** | 14.97 |  7.95 |  6.14 |  5.34 |  5.21 |  4.58 |  4.34 |  3.74 |   2.96 |
| **BPW**       | 16.00 |  8.50 |  6.56 |  5.70 |  5.57 |  4.89 |  4.64 |  4.00 |   3.16 |
| **PPL**       | 5.130 | 5.135 | 5.148 | 5.181 | 5.222 | 5.312 | 5.549 | 5.755 | 11.859 |
| **PP Speed**  |  5.99 |  6.10 |  7.17 |  7.34 |  6.65 |  6.38 |  6.00 |  6.85 |   6.43 |
| **TG Speed**  | 44.03 | 26.08 | 21.61 | 22.33 | 20.93 | 18.93 | 17.09 | 22.50 |  19.21 |

> [!NOTE]
>
> - 模型大小：单位GB
> - BPW（Bits-Per-Weight）：单位参数比特，例如Q8_0实际平均精度为8.50
> - PPL（困惑度）：以8K上下文测量（原生支持长度），数值越低越好
> - PP/TG速度：提供了Apple M3 Max（Metal）的指令处理（PP）和文本生成（TG）速度，单位ms/token，数值越低越快

## 训练与精调

### 手动训练与精调

- 使用无标注数据进行预训练：[📖预训练脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/pt_scripts_zh)
- 使用有标注数据进行指令精调：[📖指令精调脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/sft_scripts_zh)

### 指令模板

本项目Llama-3-Chinese-Instruct沿用原版Llama-3-Instruct的指令模板。以下是一组对话示例：

> <|begin_of_text|><|start_header_id|>system<|end_header_id|>
>
> You are a helpful assistant. 你是一个乐于助人的助手。<|eot_id|><|start_header_id|>user<|end_header_id|>
>
> 你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>
>
> 你好！有什么可以帮助你的吗？<|eot_id|>

### 指令数据

以下是本项目开源的部分指令数据。详情请查看：[📚 指令数据](./data)

| 数据名称                                                     | 说明                                                         | 数量 |
| ------------------------------------------------------------ | :----------------------------------------------------------- | :--: |
| [alpaca_zh_51k](https://huggingface.co/datasets/hfl/alpaca_zh_51k) | 使用gpt-3.5翻译的Alpaca数据                                  | 51K  |
| [stem_zh_instruction](https://huggingface.co/datasets/hfl/stem_zh_instruction) | 使用gpt-3.5爬取的STEM数据，包含物理、化学、医学、生物学、地球科学 | 256K |
| [ruozhiba_gpt4_turbo](https://huggingface.co/datasets/hfl/ruozhiba_gpt4_turbo) | 使用`gpt-4-turbo-2024-04-09`获取的ruozhiba问答数据           | 2449 |

## 常见问题

请在提交Issue前务必先查看FAQ中是否已存在解决方案。具体问题和解答请参考本项目 [📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/faq_zh)

```
问题1：为什么没有像一期、二期项目一样做词表扩充？
问题2：会有70B版本发布吗？
问题3：为什么指令模型不叫Alpaca了？
问题4：本仓库模型能否商用？
问题5：为什么不对模型做全量预训练而是用LoRA？
问题6：为什么Llama-3-Chinese对话效果不好？
问题7：为什么指令模型会回复说自己是ChatGPT？
问题8：Instrcut模型的v1（原版）和v2有什么区别？
```

## 免责声明

本项目基于由Meta发布的Llama-3模型进行开发，使用过程中请严格遵守Llama-3的[开源许可协议](https://github.com/meta-llama/llama3/blob/main/LICENSE)。如果涉及使用第三方代码，请务必遵从相关的开源许可协议。模型生成的内容可能会因为计算方法、随机因素以及量化精度损失等影响其准确性，因此，本项目不对模型输出的准确性提供任何保证，也不会对任何因使用相关资源和输出结果产生的损失承担责任。如果将本项目的相关模型用于商业用途，开发者应遵守当地的法律法规，确保模型输出内容的合规性，本项目不对任何由此衍生的产品或服务承担责任。


## 问题反馈

如有疑问，请在GitHub Issue中提交。礼貌地提出问题，构建和谐的讨论社区。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 提交问题请使用本项目设置的Issue模板，以帮助快速定位具体问题。
- 重复以及与本项目无关的issue会被[stable-bot](https://github.com/marketplace/stale)处理，敬请谅解。

