# Understanding How BERT Learns to Identify Edits

This is the repo with code for reproducing results from "Understanding How BERT Learns to Identify Edits".

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

1. Save attention weights for the two types of edits. This either requires a GPU or takes a long time (multiple hours).

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
