# LLaMA2中文微调

LLaMA2模型的许可证发生了变化，已允许商用，模型推出时，LLaMA2-Chat也同时推出，本人在16G推理卡上实践了微调Llama-2-7b-chat（ https://zhuanlan.zhihu.com/p/645152512 ，代码在 https://github.com/git-cloner/llama2-lora-fine-tuning ），但即使扩充了中文词表，推理效果依然不佳，回答主要以英文为主。

官方在LLaMA2模型发布时，就已开源了官方微调程序，叫做LLaMA伴侣（ https://github.com/facebookresearch/llama-recipes ），支持全量、Lora等方式微调，相对来说兼容性优于第三方的程序。

本文是在llama-recipes的基础上，修改适配显卡资源，基于Lora对LLaMA2-7b原始模型进行微调实践，结果推理效果尚可，本项目也提供了测试过程和流式接口。

- LLaMA2中文微调的效果可在Aiit-Chat查看，链接地址为： https://gitclone.com/aiit/chat/ 。


## 1、推理卡要求

16G及以上，最好有两块以上。

100多M的语料，在两块P100（16G）上微调一轮需要120小时。所以建议使用V100、4090等推理卡微调。

## 2、微调过程

### 2.1 下载代码

```bash
git clone https://github.com/git-cloner/Llama2-chinese
cd Llama2-chinese
```

### 2.2 安装虚拟环境

```bash
conda create -n llama-recipes python=3.9 -y
conda activate llama-recipes
# 因为requirements中有从github中安装的依赖，网络环境不佳，打开这两个参数可以观察进度
export GIT_TRACE=1
export GIT_CURL_VERBOSE=1
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple --trusted-host=pypi.mirrors.ustc.edu.cn
# 问题比较多的是bitsandbytes，pip install后用以下命令验证
python -m bitsandbytes
```

### 2.3 下载Llama2-7b原始模型

```bash
# 用本项目开发的下载器下载模型，可以断点续传和重连
python model_download.py --repo_id NousResearch/Llama-2-7b-hf
# 下载后的模型在 ./models\NousResearch\Llama-2-7b-hf 下
```

### 2.4 语料准备

语料采用了alpaca格式（huggingface.co中alpaca语料很多，可自行整理），个性化修改后，命名为：ft_datasets/alpaca_data.json

### 2.5 微调过程


```bash
# kill process force
pkill -9 -f llama_finetuning
# train，batch_size_training可按显存大小反复试，尽量把显存占满
# 本例是用两块P100，分别是第1、2块
# ！注意如果用两块卡，nproc_per_node是1，不是2
CUDA_VISIBLE_DEVICES=1,2 nohup torchrun --nnodes 1 --nproc_per_node 1   \
llama_finetuning.py \
--use_peft \
--peft_method lora \
--model_name ./models/NousResearch/Llama-2-7b-hf \
--use_fp16 \
--output_dir output/model \
--dataset alpaca_dataset \
--batch_size_training 40 \
--num_epochs 3 \
--quantization > train.log  2>&1 &
# check log
tail -f train.log
```

## 3、推理测试

微调一轮后，会产生peft增量模型，在output/model下，用以下命令在客户端交互测试。由于未采用流模式，一次性生成后，才能看到结果，所以速度较慢。

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --base_model './models/NousResearch/Llama-2-7b-hf' \
    --lora_weights './output/model' \
    --load_8bit 
```

## 4、流式API测试

### 4.1 开启API服务

```bash
# 可以用4bit或8bit量化方式或半精度装入模型测试
# --load_4bit  需要约6G显存
# --load_8bit  需要9G显存
# 半精度  需要13G显存
CUDA_VISIBLE_DEVICES=0 nohup python -u api_stream.py \
--load_4bit > api_stream.log  2>&1 &
tail -f api_stream.log
```

### 4.2 测试API

```bash
# 多次发POST请求，直到返回的response中包含[stop]后停止调用
curl -X POST "http://127.0.0.1:8000/stream" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你好", "history": []}'
```

## 5、模型合并

```bash
python inference/hf-text-generation-inference/merge_lora_weights.py \
--base_model ./models/NousResearch/Llama-2-7b-hf \
--peft_model output/model \
--output_dir output/merged_model_output
```

## 6、存在问题

- 尽量做全量或半精度微调，Lora的效果一般
- 本项目中，由于算力限制，max_token_size设置的比较小（256），精度也低（4bit）所以生成的可能不完整
- 语料不宜过多，但要求质量要高，5万多条（51K）效果好