# Pretraining Language Models with Human Preferences

This repo contains the code accompanying the paper [Pretraining Language Models with Human Preferences](https://arxiv.org/abs/2302.08582). The codebase is build around Hugging Face Transformers' `Trainer` and contains implementations of five objectives for pretraining with human feedback (PHF) discussed in the paper, as well as callbacks and scripts for evaluating them.

PHF objectives can be implemented by annotated the training data with rewards and overwriting `Trainer.compute_loss` to use them as additional training signal. Rewards are provided by an instance of `apo.scorers.Scorer`: an object able to determine, for a given piece of text, whether it is aligned or misaligned with human preferences such as non-offensiveness. The scorer is also used for evaluating samples from PHF-trained LMs.

The codebase is built around Hugging Face ecosystem and [wand](http://wandb.ai) (for monitoring and experiment management). 

## Quickstart

We assume Python 3.9+. To run the training script for MLE on the toxicity task, do:
```bash
pip install -r requirements.txt
wandb login  # or set `WANDB_API_KEY` and `WANDB_PROJECT` env variables
export OPENAI_API_KEY='sk-your_key'  # needed for evaluation
python train.py --task configs/toxicity/pretrain.yml --method configs/toxicity/mle.yml
```

### Configuration

The `train.py` scripts requires paths to two config files: for task and for method. Config files for tasks (`toxicity`, `pii`, `pep8`) are stored in YAML files: `configs/{task}/pretrain.yml` (for pretraining experiments) and `configs/{task}/finetuning.yml` (for finetuning). Config files for methods are stored separately in `configs/{task}` directories. Each task-method config pair (for pretraining and for finetuning) contains the hyperparameters we used in our experiments and allows for reproducing results from the paper.

Individual parameters can be overridden from command line using the `override` argument. For instance:
```bash
python train.py --task configs/toxicity/pretrain.yml --method configs/toxicity/mle.yml --override training.per_device_train_batch_size=8
```

## Tasks

| Name        | Config files       | Training data                                                                                                                                 | Scorer            | Description
| ----------- |--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------| --------          | --------
| Toxicity    | `configs/toxicity` | [`tomekkorbak/pile-detoxify`](https://huggingface.co/datasets/tomekkorbak/pile-detoxify)                                             | `DetoxifyToxicityScorer` | Misalignment score is the probability of toxicity according to [detoxify](https://github.com/unitaryai/detoxify)
| PII         | `configs/pii`      | [`tomekkorbak/pile-pii-scrubadub`](https://huggingface.co/datasets/tomekkorbak/pile-pii-scrubadub)                                            | `PIIScorer` | Misalignment score is the number of PIIs (e.g. names, URLs) per character, according to [scrubadub](https://github.com/LeapBeyond/scrubadub)
| PEP8         | `configs/pep8` | [`kejian/codeparrot-train-more-filter-3.3b-cleaned`](https://huggingface.co/datasets/kejian/kejian/codeparrot-train-more-filter-3.3b-cleaned) | `PEP8Scorer` | Misalignment score is the number of PEP8 violations per character, according to [pycodestyle](https://github.com/PyCQA/pycodestyle)

## Objectives 

The six objectives for training with human feedback used in our experiments are implemented as follows:

| Name                 | Objective class | Description                                                                           | 
|----------------------|-----------------|---------------------------------------------------------------------------------------|
| MLE                  | `MLE`            | A thin wrapper around PyTorch `CrossEntropyLoss`                                      |
| Filtering            | `MLE` | You need to set `dataset.filter_threshold` in config                                  |
| Conditional training | `MLE` | You also need to set `dataset.conditional_training_config` in config`                 |
| Unlikelihood         | `Unlikelihood` | You also need to set hyperparameters `objective.score_threshold` and `objective.alpha` |
| AWR                  | `AWR` | You also need to set hyperparameters `objective.alpha` and `objective.beta`           |
| RWR                  | `AWR` | A special case of AWR with `objective.alpha=1`                                        |   



## Metrics

On each evaluation step, `apo.callbacks.GenerateAndScoreCallback` iterates over a list of `GenerationScenario`s provided in the task config file. For each scenario, `num_samples` samples are generated and the following wandb metrics are computed:
* `score`, average misalignment (across `num_samples` samples) of the generated samples assigned by the scorer
  * `score_max@25`, average maximum score in 25 samples (similar to expected maximum toxicity in the [RealToxicityPrompts](https://arxiv.org/abs/2009.11462) paper)
* `current_samples`, a [`wandb.Table`](https://docs.wandb.ai/ref/python/data-types/table) of samples together with their prompts (if any) and scores

In addition to scoring LM samples, we use `apo.callbacks.KLGPT3Callback` to estimate KL of the current LM from GPT-3. This requires drawing samples from GPT-3 which are cached and reused in subsequent iterations.
                                                                    |


## Codebase structure

```bash
.
├── apo
│   ├── callbacks.py  # callbacks implementing the evaluation pipeline 
│   ├── dataset_wrappers.py  # an iterable for streaming blocks of tokens for training
│   ├── kl_gpt3.py  # logic for measuring KL from GPT-3
│   └── metrics.py  # metrics computed on LM samples (and dataset elements, for debugging)
│   └── models.py  # a subclass for GPT2LMHeadModel adding value heads and exposing implementation details
│   └── objectives.py  # classes implementing loss functions
│   ├── scorer_utils.py
│   ├── scorers.py  # classes for scoring LM samples and dataset elements
│   └── trainer.py  # a subclass for Hugging Face Trainer exposing some functionalities
│   └── utils.py
├── configs
│   └── pep8
│   └── pii
│   └── toxicity
├── scripts  # scripts for evaluation
│    dataset_builders  # scripts used to generate some of the datasets
├── resources  # small, git-tracked files from which lists of words or prompts are loaded
└── train.py  # the main training script
```

## Citing

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.08582,
  doi = {10.48550/ARXIV.2302.08582},
  url = {https://arxiv.org/abs/2302.08582},
  author = {Korbak, Tomasz and Shi, Kejian and Chen, Angelica and Bhalerao, Rasika and Buckley, Christopher L. and Phang, Jason and Bowman, Samuel R. and Perez, Ethan},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Pretraining Language Models with Human Preferences},
  publisher = {arXiv},  
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
