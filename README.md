<div align="center">
  <h1>VideoOdyssey</h1>
  <h2>A Benchmark for Ultra-Long-Context and Omni-Modal Video Understanding</h2>
</div>

<p align="center">
  <b>Haichen He</b><sup>1,*</sup>, <b>Jiayi Zhou</b><sup>1,*</sup>, <b>Sifeng Shang</b><sup>1</sup>, <b>Yihan Hu</b><sup>3</sup>, <b>Yuanhan Zhang</b><sup>2</sup>, <b>Kaiyang Zhou</b><sup>1,†</sup>
  <br><br>
  <sup>*</sup>Equal Contribution &nbsp;&nbsp;&nbsp;&nbsp; <sup>†</sup>Corresponding Author
  <br>
  <sup>1</sup>Hong Kong Baptist University &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>S-Lab, Nanyang Technological University
  <br>
  <sup>3</sup>GVC Lab, Great Bay University
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-VideoQA-red" alt="VideoQA"> 
  <img src="https://img.shields.io/badge/Task-AudioVideo--QA-red" alt="AudioVideo-QA"> 
  <img src="https://img.shields.io/badge/Task-Multi--Modal-red" alt="Multi-Modal"> 
  <img src="https://img.shields.io/badge/Dataset-VideoOdyssey-blue" alt="VideoOdyssey"> 
</p>

<font size=5><div align='center' > [[🍎 Project Page](https://github.com/maifoundations/VideoOdyssey)] [[📖 Arxiv Paper](https://arxiv.org/abs/xxxx.xxxxx)] [[📊 Dataset](https://huggingface.co/datasets/maifoundations/VideoOdyssey)] [[🏆 Leaderboard](https://videoodyssey-project.github.io/#leaderboard)] </div></font>


<p align="center">
    <img src="./asset/example.png" width="100%" height="100%">
</p>


## 🔥 News
* **`2026-05-18`** 🌟 We release VideoOdyssey, a benchmark for ultra-long-context and omni-modal video understanding.


## 👀 VideoOdyssey Overview

[//]: # (Insert a concise version of your abstract here.)
VideoOdyssey introduces a comprehensive evaluation benchmark specifically designed to assess ultra-long-context and omni-modal capabilities in modern Multimodal Large Language Models (MLLMs).

Our dataset and evaluation pipeline are built upon three core characteristics:
* **[Characteristic 1 Title]**: [Brief description of the first key feature, e.g., Ultra-long context duration.]
* **[Characteristic 2 Title]**: [Brief description of the second key feature, e.g., Omni-modal integration.]
* **[Characteristic 3 Title]**: [Brief description of the third key feature, e.g., Comprehensive CCL levels.]

<p align="center">
    <img src="./asset/benchmark_statistics.png" width="100%" height="100%">
</p>


## 📐 Dataset Examples

<p align="center">
    <img src="./asset/example.png" width="100%" height="100%">
</p>


## 🔍 Dataset

**License**:
> VideoOdyssey is under the CC-BY-NC-SA-4.0 license. VideoOdyssey is only used for academic research. Commercial use in any form is prohibited. We do not own the copyright of any raw video files. If there is any infringement, please contact us and we will remove it immediately.

You can download the dataset, subtitles, and annotations from [Hugging Face](https://huggingface.co/datasets/your-repo).


## 🔮 Evaluation

Once you have downloaded the datasets, subtitles, and annotations, you can start the evaluation.

### Prompts
Depending on your evaluation setting, use the following prompt templates:
* **Video**: `[Insert Prompt Template Here]`
* **Video + Subtitles**: `[Insert Prompt Template Here]`
* **Video + Audio**: `[Insert Prompt Template Here]`

### Use Certificate Window (CW)
By default, we evaluate models under the **w/o CW** setting, which means the complete, uncut video is input into the model. 

However, if you want to evaluate under the **w/ CW** setting, you need to slice the video according to the `time_reference` interval specified for each question. 
We provide a [script as an example](https://github.com/your-repo/cw-slice-script) to handle this slicing.

* If you want to test on **VideoOdyssey-V** under the **w/o CW** setting (inputting video results only), you can run the following code:
```bash
python run_eval.py \
    --dataset VideoOdyssey-V \
    --setting wo_cw \
    --input_type video \
    --results_dir ./results
