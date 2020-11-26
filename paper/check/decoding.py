import argparse

from .. import csv_merge, data, tokenizers, util
from ..structures import HyperParameters


def check_special_tokens(original: str, decoded: str) -> None:
    for tok in tokenizers.AESW_TOKENS:
        assert (tok in original) == (tok in decoded), f"'{original}' '{decoded}'"


def check_encoding_decoding(config: HyperParameters) -> None:
    data_files = [config.val_file, config.test_file]

    tokenizer = tokenizers.get_tokenizer(config)

    for sentence, label in util.my_tqdm(csv_merge.sorted_csv_files_reader(data_files)):
        ids_list, _ = tokenizers.encode_batch([sentence], tokenizer)
        decoded = tokenizers.decode(ids_list[0].tolist(), tokenizer)
        check_special_tokens(sentence, decoded)


def check_decoding(config: HyperParameters) -> None:
    tokenizer = tokenizers.get_tokenizer(config)

    print("Preprocessing data...")
    data.prepare_val_data(config, tokenizer)

    loader = data.get_val_dataloader(config)

    print("Decoding data...")
    for batch in util.my_tqdm(loader):
        original_ids = [ids.tolist() for ids in batch[0]]
        sentences = [tokenizers.decode(ids, tokenizer) for ids in original_ids]

        for sent in sentences:
            if "_MATH_" in sent:
                math_seen = True
            if "_MATHDISP_" in sent:
                mathdisp_seen = True

    assert math_seen, "didn't see any '_MATH_'"
    assert mathdisp_seen, "didn't see any '_MATHDISP_'"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config", help="location of experiment .toml config file")
    args = parser.parse_args()

    config = HyperParameters(args.config)
    check_decoding(config)
    check_encoding_decoding(config)
