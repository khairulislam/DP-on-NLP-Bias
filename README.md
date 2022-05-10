# Evaluating Bias and Robustness of Differentially Private NLP Models

## Abstract

In this work we evaluate how differentially private training impacts the NLP models based on overall performance, bias and robustness. We experiment with different privacy budgets, bias metrics, adversarial attacks to perform the evaluation. Our experiments show that Differential Privacy (DP) can mitigate the bias of the NLP model at the cost of reduced overall performance.

## Dataset

Complete Jigsaw unintented bias data [all_data.csv](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data?select=all_data.csv)

## Evaluation Metrics

* [Bias AUC](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation)
* [EOdds](https://arxiv.org/pdf/2106.10826.pdf)

## Model

Pretrained [Bert-small](https://huggingface.co/prajjwal1/bert-small). We only train the last three layers of the model.

## Privacy Engine

[Opacus](https://opacus.ai/). Used delta=0.05, epsilon=0.1, 0.2, 0.25, 0.3, 0.35, 0.5.

## Expertimentation Setup

* [Tokenize jigsaw comments](src/tokenize-jigsaw-comments.ipynb) is used to undersample the train data. Then tokenize both the undersampled train set and original test set.
* [Tuning on jigsaw unintended bias](src/tuning-on-jigsaw-unintended-bias.ipynb) notebook is used to run the non-private trainings.
* [Private tuning on jigsaw unintended bias](src/private-tuning-on-jigsaw-unintended-bias.ipynb) notebook is used to run the private trainings.
* [benchmark](src/benchmark.ipynb) benchmarks the experimentation results based on the target identity columns. For now only gender and race are used.

## Others

### Tutorials

* [Bias and Fairness in Natural Language Processing](http://web.cs.ucla.edu/~kwchang/talks/emnlp19-fairnlp/)
* [Fairness and Bias Mitigation](https://guide.allennlp.org/fairness)

### Notebooks

#### Jigsaw Unintended Bias in Toxicity Classification

* [Kaggle competition](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/code?competitionId=12500)
* [Intro to AI Ethics - Identifying Bias in AI](https://www.kaggle.com/code/georgezoto/intro-to-ai-ethics-identifying-bias-in-ai)
* [Benchmark Kernel](https://www.kaggle.com/code/dborkan/benchmark-kernel/notebook)

#### Kaggle

* [Identifying Bias in AI](https://www.kaggle.com/code/alexisbcook/identifying-bias-in-ai/tutorial)