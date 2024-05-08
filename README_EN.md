[**🇨🇳Chinese**](./README.md) | [**🌐English**](./README_EN.md) | [**📖Documentation**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki) | [**❓Issues**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/issues) | [**💬Discussions**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/discussions) | [**⚔️Arena**](http://llm-arena.ymcui.com/)

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

This project is developed based on Meta's newly released next-generation open-source large language model [Llama-3](https://github.com/facebookresearch/llama3) and is the third generation of the Chinese-LLaMA-Alpaca open-source LLM series ([1st gen](https://github.com/ymcui/Chinese-LLaMA-Alpaca), [2nd gen](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)). This project has open-sourced **the Llama-3-Chinese base model and the Chinese Llama-3-Chinese-Instruct instruction-tuned large model**. These models use large-scale Chinese data for continual pre-training on the original Llama-3, and are fine-tuned with selected instruction data to further enhance Chinese basic semantic and instruction understanding capabilities, significantly improving performance compared to the second-generation models.

#### Main Content

- 🚀 Open-source Llama-3-Chinese base model and Llama-3-Chinese-Instruct instruction model
- 🚀 Released pre-training scripts and instruction fine-tuning scripts, allowing users to further train or fine-tune the model as needed
- 🚀 Released alpaca_zh_51k, stem_zh_instruction, ruozhiba_gpt4_turbo instruction data
- 🚀 Provides a tutorial for quickly quantizing and deploying large models locally using a personal computer's CPU/GPU
- 🚀 Supports [🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [vLLM](https://github.com/vllm-project/vllm), [Ollama](https://ollama.com) and other Llama-3 ecosystems

----

[Chinese Mixtral](https://github.com/ymcui/Chinese-Mixtral) | [Chinese LLaMA-2 & Alpaca-2 Large Models](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) | [Chinese LLaMA & Alpaca Large Models](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | [Multimodal Chinese LLaMA & Alpaca Large Models](https://github.com/airaria/Visual-Chinese-LLaMA-Alpaca) | [Multimodal VLE](https://github.com/iflytek/VLE) | [Chinese MiniRBT](https://github.com/iflytek/MiniRBT) | [Chinese LERT](https://github.com/ymcui/LERT) | [Chinese-English PERT](https://github.com/ymcui/PERT) | [Chinese MacBERT](https://github.com/ymcui/MacBERT) | [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [Chinese XLNet](https://github.com/ymcui/Chinese-XLNet) | [Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [Knowledge Distillation Tool TextBrewer](https://github.com/airaria/TextBrewer) | [Model Pruning Tool TextPruner](https://github.com/airaria/TextPruner) | [Distillation and Pruning Integrated GRAIN](https://github.com/airaria/GRAIN)


## News

**[2024/05/08] Release Llama-3-Chinese-8B-Instruct-v2, which is directly tuned on  [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) with 5M instructions. For details, see: [📚Version 2.0 Release Log](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/releases/tag/v2.0)**

[2024/05/07] Add pre-training and SFT scripts. For details, see: [📚Version 1.1 Release Log](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/releases/tag/v1.1)

[2024/04/30] Released the Llama-3-Chinese-8B base model and Llama-3-Chinese-8B-Instruct instruction model. For details, see: [📚Version 1.0 Release Log](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/releases/tag/v1.0)

[2024/04/19] 🚀 Officially launched the Chinese-LLaMA-Alpaca-3 project


## Content Guide
| Section                              | Description                                                   |
| ------------------------------------ | ------------------------------------------------------------- |
| [💁🏻‍♂️Model Introduction](#model-introduction) | Briefly introduces the technical features of the models related to this project |
| [⏬Model Download](#model-download)  | Download addresses for the Chinese Llama-3 large models      |
| [💻Inference and Deployment](#inference-and-deployment) | Describes how to quantize the model and deploy it using a personal computer to experience the large model |
| [💯Model Performance](#model-performance) | Introduces the effects of the model on some tasks             |
| [📝Training and Fine-Tuning](#training-and-fine-tuning) | Introduces how to train and fine-tune the Chinese Llama-3 large models |
| [❓Frequently Asked Questions](#frequently-asked-questions) | Replies to some common questions                              |


## Model Introduction

This project has launched the Chinese open-source large models Llama-3-Chinese and Llama-3-Chinese-Instruct based on Meta Llama-3. The main features are as follows:

#### 📖 Uses the Original Llama-3 Vocabulary

- Llama-3 has significantly expanded its vocabulary from 32K to 128K and switched to a BPE vocabulary.
- Preliminary experiments have shown that the encoding efficiency of the Llama-3 vocabulary is comparable to our expanded vocabulary in [Chinese LLaMA-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2), with an efficiency of about 95% based on encoding efficiency tests on Wikipedia data.
- Based on our experience and experimental conclusions with [Chinese Mixtral](https://github.com/ymcui/Chinese-Mixtral) [^1], we **did not expand the vocabulary further**.

[^1]: [Cui and Yao, 2024. Rethinking LLM Language Adaptation: A Case Study on Chinese Mixtral](https://arxiv.org/abs/2403.01851)

#### 🚄 Extended Context Length from 4K in the Second Generation to 8K

- Llama-3 has increased the native context window length from 4K to 8K, allowing for further processing of longer context information.
- Users can also use methods like PI, NTK, and YaRN to extend the model's long context capabilities to support longer text processing.

#### ⚡ Uses Grouped Query Attention Mechanism

- Llama-3 adopts the Grouped Query Attention (GQA) mechanism used in the large parameter version of Llama-2, which further enhances the model's efficiency.

#### 🗒 New Instruction Template

- Llama-3-Instruct uses a new instruction template, which is not compatible with Llama-2-chat; it should be used strictly following the official instruction template. (See [instruction template](#instruction-template))

## Model Download

### Model Selection Guide

Here's a comparison of the models in this project and recommended usage scenarios. **For chat interactions, please choose the Instruct version.**

| Comparison Item         | Llama-3-Chinese-8B                      | Llama-3-Chinese-8B-Instruct             |
| ----------------------- | :-------------------------------------: | :----------------------------------------------: |
| Model Type              | Base Model                              | Instruction/Chat Model (similar to ChatGPT)      |
| Model Size              | 8B                                      | 8B                                               |
| Training Type           | Causal-LM (CLM)                         | Instruction Fine-Tuning                          |
| Training Method         | LoRA + Full emb/lm-head                 | LoRA + Full emb/lm-head                          |
| Initial Model           | [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | v1: Llama-3-Chinese-8B<br/>v2: [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| Training Corpus         | Unlabeled general corpus (approx. 120GB) | Labeled instruction data (approx. 5 million entries) |
| Vocabulary Size         | Original vocabulary (128,256)           | Original vocabulary (128,256)                    |
| Supported Context Length | 8K                                      | 8K                                               |
| Input Template          | Not required                            | Requires Llama-3-Instruct template               |
| Applicable Scenarios    | Text continuation: Given a context, let the model generate the following text | Instruction understanding: Q&A, writing, chatting, interaction, etc. |


### Download Links

| Model Name                                          |                         Full Version                         |                         LoRA Version                         |                         GGUF Version                         |
| --------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **Llama-3-Chinese-8B-Instruct-v2**<br/>(chat model) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2-lora)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2-lora)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2-lora) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2-gguf)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v2-gguf) |
| **Llama-3-Chinese-8B-Instruct**<br/>(chat model)    | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-lora)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-lora)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-lora) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-instruct-gguf)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-gguf) |
| **Llama-3-Chinese-8B**<br/>(base model)             | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-lora)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-lora)<br/>[[wisemodel]](https://wisemodel.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-lora) | [[🤗Hugging Face]](https://huggingface.co/hfl/llama-3-chinese-8b-gguf)<br/> [[🤖ModelScope]](https://modelscope.cn/models/ChineseAlpacaGroup/llama-3-chinese-8b-gguf) |


Model Type Description:

- **Full Model**: Can be used directly for training and inference, no other merging steps required.
- **LoRA Model**: Needs to be merged with the original base model to convert into a full version, merging steps: [**💻 Model Merging Steps**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/model_conversion_en)
  - v1 base model:  [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 
  - v2 base model:  [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 
- **GGUF Model**: Quantization format released by [llama.cpp](https://github.com/ggerganov/llama.cpp), compatible with common large model inference tools like ollama, recommended for users who only need to perform inference deployment. The model name with `-im` suffix is generated with important matrix, which has generally better performance.
> [!NOTE]
> If HF access is blocked, consider using mirror sites (like [hf-mirror.com](hf-mirror.com)), please find the specific methods and solutions on your own.

## Inference and Deployment

The models in this project primarily support the following quantization, inference, and deployment methods. Please refer to the corresponding tutorials for detailed information.

| Tool | Features | CPU | GPU | Quantization | GUI | API | vLLM | Tutorial |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Rich GGUF quantization options and efficient local inference | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/llamacpp_en) |
| [🤗transformers](https://github.com/huggingface/transformers) | Native transformers inference interface | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/inference_with_transformers_en) |
| [Imitation OpenAI API Calls](https://platform.openai.com/docs/api-reference) | Server demo with an interface similar to OpenAI API | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/openai_api_en) |
| [text-generation-webui](https://github.com/oobabooga/text-generation-webui) | Front-end Web UI deployment method | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/text-generation-webui_en) |
| [LM Studio](https://lmstudio.ai) | Multi-platform chat software with interface | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/lmstudio_en) |
| [Ollama](https://ollama.com) | Local large model inference | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | [[link]](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/ollama_en) |


## Model Performance

To evaluate the effectiveness of the related models, this project conducted both generative performance evaluations and objective performance evaluations (NLU type), assessing the large models from different perspectives. Users are recommended to test on tasks of their interest and choose models suitable for those tasks.

### Generative Performance Evaluation

- This project has launched an online model battle platform, modeled after the [Fastchat Chatbot Arena](https://chat.lmsys.org/?arena), where users can browse and evaluate the quality of model responses. The battle platform provides metrics such as win rates and Elo scores, and allows viewing the win rates between different models. **⚔️ Model Arena: [http://llm-arena.ymcui.com](http://llm-arena.ymcui.com/)**
- The examples directory provides output samples of Llama-3-Chinese-8B-Instruct and Chinese-Mixtral-Instruct, and compares scores using GPT-4-turbo, **with Llama-3-Chinese-8B-Instruct averaging a score of 8.1 and  Chinese-Mixtral-Instruct averaging 7.8**. **📄 Output Sample Comparison: [examples](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/blob/main/examples)**
- This project has joined the Machine Heart SOTA! Model platform, with online experiences to be implemented later: https://sota.jiqizhixin.com/project/chinese-llama-alpaca-3


### Objective Performance Evaluation

#### C-Eval

[C-Eval](https://cevalbenchmark.com) is a comprehensive Chinese fundamental model evaluation suite, with its validation and test sets comprising 1.3K and 12.3K multiple-choice questions respectively, covering 52 subjects. For C-Eval inference code, please refer to this project: [📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/ceval_en)

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

[CMMLU](https://github.com/haonan-li/CMMLU) is another comprehensive Chinese evaluation dataset specifically designed to assess language models' knowledge and reasoning capabilities in a Chinese context, covering topics from basic subjects to advanced professional levels, with a total of 11.5K multiple-choice questions. For CMMLU inference code, please refer to this project: [📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/cmmlu_en)

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

[MMLU](https://github.com/hendrycks/test) is an English evaluation dataset for assessing natural language understanding capabilities, one of the main datasets used today for evaluating large models' capabilities, with its validation and test sets comprising 1.5K and 14.1K multiple-choice questions respectively, covering 57 subjects. For MMLU inference code, please refer to this project: [📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/mmlu_en)

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

[LongBench](https://github.com/THUDM/LongBench) is a benchmark for evaluating large models' long-text understanding capabilities, composed of 6 categories and 20 different tasks. Most tasks have an average length between 5K-15K, totaling approximately 4.75K test data entries. Below are the evaluation results of this project's models on these Chinese tasks (including code tasks). For LongBench inference code, please refer to this project: [📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/longbench_en)

| Models | Single-doc QA | Multi-doc QA | Summarization | Few-Shot Learning | Code | Synthesis | Average |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
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

[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is an LLM benchmark (English) brought by HuggingFaceH4 team, including ARC, HellaSwag, MMLU, TruthfulQA, Winograde, GSM8K datasets. Below are the evaluation results of this project's models.

| Models                                                       |  ARC  | HellaS | MMLU  |  TQA  | WinoG | GSM8K | Average |
| ------------------------------------------------------------ | :---: | :----: | :---: | :---: | :---: | :---: | :-----: |
| **Llama-3-Chinese-8B-Instruct-v2**                           | 62.63 | 79.72  | 66.48 | 53.93 | 76.72 | 60.58 |  66.68  |
| **Llama-3-Chinese-8B-Instruct**                              | 61.26 | 80.24  | 63.10 | 55.15 | 75.06 | 44.43 |  63.21  |
| **Llama-3-Chinese-8B**                                       | 55.88 | 79.53  | 63.70 | 41.14 | 77.03 | 37.98 |  59.21  |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 60.75 | 78.55  | 67.07 | 51.65 | 74.51 | 68.69 |  66.87  |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | 59.47 | 82.09  | 66.69 | 43.90 | 77.35 | 45.79 |  62.55  |
| [Chinese-Mixtral-Instruct](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 67.75 | 85.67  | 71.53 | 57.46 | 83.11 | 55.65 |  70.19  |
| [Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral) (8x7B) | 67.58 | 85.34  | 70.38 | 46.86 | 82.00 | 0.00  |  58.69  |

*Note: MMLU resutls are different from the one that reported in our repo, as the evaluation scripts differ.*

### Quantitative Performance Evaluation

Under llama.cpp, the quantization performance of Llama-3-Chinese-8B (base model) was tested, as shown in the table below. The actual speed is slightly slower than the second-generation Llama-2-7B.

|               |    F16 |   Q8_0 |   Q6_K |   Q5_K |   Q5_0 |   Q4_K |   Q4_0 |   Q3_K |    Q2_K |
| ------------- | -----: | -----: | -----: | -----: | -----: | -----: | -----: | -----: | ------: |
| **Size (GB)** |  14.97 |   7.95 |   6.14 |   5.34 |   5.21 |   4.58 |   4.34 |   3.74 |    2.96 |
| **BPW**       |  16.00 |   8.50 |   6.56 |   5.70 |   5.57 |   4.89 |   4.64 |   4.00 |    3.16 |
| **PPL**       | 5.130 | 5.135 | 5.148 | 5.181 | 5.222 | 5.312 | 5.549 | 5.755 | 11.859 |
| **PP Speed**  |   5.99 |   6.10 |   7.17 |   7.34 |   6.65 |   6.38 |   6.00 |   6.85 |    6.43 |
| **TG Speed**  |  44.03 |  26.08 |  21.61 |  22.33 |  20.93 |  18.93 |  17.09 |  22.50 |   19.21 |

> [!NOTE]
>
> - Model size: in GB
> - BPW (Bits-Per-Weight): Per-parameter bit, for example, Q8_0 actual average precision is 8.50
> - PPL (Perplexity): Measured with an 8K context (natively supported length), lower values are better
> - PP/TG speed: Provides instruction processing (PP) and text generation (TG) speeds for the Apple M3 Max (Metal), in ms/token, lower values are faster

## Training and Fine-Tuning

### Manual Training and Fine-Tuning

- Pre-training with unlabeled data: [📖Pre-training Scripts Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/pt_scripts_en)
- Fine-tuning with labeled data for instructions: [📖Instruction Fine-Tuning Scripts Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/sft_scripts_en)

### Instruction template

Our Llama-3-Chinese-Instruct adopts original instruction template of Llama-3-Instruct. The following is a chat example.

> <|begin_of_text|><|start_header_id|>system<|end_header_id|>
>
> You are a helpful assistant. 你是一个乐于助人的助手。<|eot_id|><|start_header_id|>user<|end_header_id|>
>
> 你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>
>
> 你好！有什么可以帮助你的吗？<|eot_id|>

### Data

Below are some of the command data made open source for this project. For more details, please see: [📚 Instruction Data](./data)

| Data Name                                                    | Description                                                  | Quantity |
| ------------------------------------------------------------ | :----------------------------------------------------------- | :------: |
| [alpaca_zh_51k](https://huggingface.co/datasets/hfl/alpaca_zh_51k) | Alpaca data translated using gpt-3.5                         |   51K    |
| [stem_zh_instruction](https://huggingface.co/datasets/hfl/stem_zh_instruction) | STEM data scraped using gpt-3.5, including physics, chemistry, medicine, biology, earth sciences |   256K   |
| [ruozhiba_gpt4_turbo](https://huggingface.co/datasets/hfl/ruozhiba_gpt4_turbo) | ruozhiba Q&A data obtained using `gpt-4-turbo-2024-04-09`    |   2449   |

## Frequently Asked Questions

Please check the FAQ to see if a solution already exists before submitting an issue. For specific questions and answers, refer to the project's [📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3/wiki/faq_en)

```
Question 1: Why is there no vocabulary expansion like in phases one and two?
Question 2: Will there be a 70B version released?
Question 3: Why is the instruction model no longer called Alpaca?
Question 4: Can the models from this repository be used commercially?
Question 5: Why not perform full pre-training instead of using LoRA?
Question 6: Why is the conversational performance of Llama-3-Chinese not good?
Question 7: Why does the instruction model reply saying it is ChatGPT?
Question 8: What are the differences between v1 and v2 of the Instruct model?
```

## Disclaimer

This project is developed based on Meta's Llama-3 model. Please strictly adhere to the Llama-3 [open-source license agreement](https://github.com/meta-llama/llama3/blob/main/LICENSE) during use. If using third-party code, comply with the relevant open-source licenses. The accuracy of the model-generated content may be affected by computational methods, random factors, and loss of quantization precision, hence, no guarantees are provided regarding the accuracy of model outputs, nor will any liability be accepted for losses resulting from the use of related resources and outputs. If using the models for commercial purposes, developers must comply with local laws and regulations to ensure the legality of the model outputs. No responsibility will be taken for any products or services derived from this project.


## Feedback

If you have questions, please submit them in the GitHub Issues. Ask politely and help build a harmonious discussion community.

- Before submitting an issue, check if the FAQ addresses your question and consider reviewing past issues that might solve your problem.
- When submitting an issue, please use the project's issue template to help quickly identify specific problems.
- Duplicate or irrelevant issues will be handled by [stable-bot](https://github.com/marketplace/stale), please understand.

