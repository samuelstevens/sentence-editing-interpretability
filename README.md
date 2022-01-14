# An Investigation of Language Model Interpretability via Sentence Editing

This is the repo with code for reproducing results from "An Investigation of Language Model Interpretability via Sentence Editing" ([Arxiv link](https://arxiv.org/abs/2011.14039))

## Data

The rationales can be found in `data-versioned/rationales/*.json.gz`. Each example is one line and contains a single JSON example. `words` is a tokenized version of the sentence, ready to be fed to a BERT-based model. `best_words` is an un-ordered list of words that make up the rationale.

## Reproducing Results

### Environment

```sh
python3 -m venv virtualenv
. ./virtualenv/bin/activate # unix
pip install -r requirements.txt
```

### Data

- You'll need to download the AESW data from [the challenge website](http://textmining.lt/aesw/aesw2016down.html), save it to `./data-unversioned/aesw`, and unzip it.

### Preprocessing

To preprocess the train, vaidation and test datasets for BERT:

```sh
python -m paper.aesw_to_sentences train,val,test
python -m paper ./experiments/bert_base_aesw_32_1e6/params.toml --preprocess train,val,test
```

### Fine-Tuning

To fine-tune BERT on the AESW task:

```sh
python -m paper ./experiments/bert_base_aesw_32_1e6/params.toml --main-loop

# If the training is picking back up:
python -m paper ./experiments/bert_base_aesw_32_1e6/params.toml --main-loop -continuing
```

### Inference

Once fine-tuned, run inference on val and test sets:

```sh
export CHECKPOINT=./models/<some hash here>.pt
python -m paper ./experiments/bert_large_aesw_16_1e6/params.toml --inference-val $CHECKPOINT
python -m paper ./experiments/bert_large_aesw_16_1e6/params.toml --inference-test $CHECKPOINT
```

### Interpretability

1. Save attention weights for the two types of edits. This either requires a GPU or takes a long time (multiple hours). You must also edit the `load_*()` functions in `paper/interpret/run.py` to specify which model you want to load.

```sh
python -m paper.interpret --weight-types=spelling,delete
```

2. Use attention weights to calculate similarity scores and accuracy. Once the attention weights have been calculated, this step is fast even without a GPU.

```sh
python -m paper.interpret --eval-types=spelling,delete
```

3. Create plots.

```sh
python -m paper.interpret.plot # --interactive if you want to show them on screen
```

4. Create individual plots comparing each model's attention on a single sentence:

```sh
python -m paper.interpret.compare --all-models "This allows us to observe Saturn's moons."
python -m paper.interpret.compare --all-models "(We'll represent a signature as an encrypted message digest):"
python -m paper.interpret.compare --finetuning "The algorithm descripted in the previous sections has several advantages."
```

## Citation

If you use this software or data, please cite our paper and the original AESW paper:

```
@article{Stevens_An_Investigation_of,
  author = {Stevens, Samuel and Su, Yu},
  journal = {BlackboxNLP 2021},
  title = {{An Investigation of Language Model Interpretability via Sentence Editing}}
}

@inproceedings{aesw,
  title = {A Report on the Automatic Evaluation of Scientific Writing Shared Task},
  author = {Daudaravicius, Vidas and Banchs, Rafael E. and Volodina, Elena and Napoles, Courtney},
  booktitle = {Proceedings of the 11th Workshop on Innovative Use of {NLP} for Building Educational Applications},
  month = jun,
  year = {2016},
  address = {San Diego, CA},
  publisher = {Association for Computational Linguistics},
  url = {https://www.aclweb.org/anthology/W16-0506},
  doi = {10.18653/v1/W16-0506},
  pages = {53--62}
}
```
