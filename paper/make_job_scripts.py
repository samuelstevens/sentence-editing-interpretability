"""

"""
import argparse
import os
import shutil
from typing import Dict, List


def make_copy_command(config_path: str) -> str:
    return f'python -m paper {config_path} --copy-models "$TMPDIR"'


def write_script(
    name: str, copy_train_command: str, experiment_folder: str, script_filename: str,
) -> None:

    config_path = f"./experiments/{experiment_folder}/params.toml"

    script_text = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1 --ntasks-per-node=48 --gpus-per-node=2
#SBATCH --account=PAS1769

set -x

# shellcheck disable=SC2064
trap "cd $SLURM_SUBMIT_DIR;mkdir $SLURM_JOB_ID;cp -R $TMPDIR/* $SLURM_JOB_ID;exit" TERM

module load python/3.7-2019.10

cd "$TMPDIR" || exit

python --version
python -m venv venv
# shellcheck disable=SC1091
. ./venv/bin/activate

pip install torch torchvision transformers==3.0.2 toml typing-extensions

cd "$SLURM_SUBMIT_DIR" || exit

cp aesw-dev.csv "$TMPDIR"
{copy_train_command}
cp -R paper "$TMPDIR"
cp -R experiments "$TMPDIR"
mkdir -p "$TMPDIR/models"
{make_copy_command(config_path)}

cd "$TMPDIR" || exit

/usr/bin/time python -m paper {config_path} --preprocess train,val

/usr/bin/time python -m paper {config_path} --main-loop

cd "$SLURM_SUBMIT_DIR" || exit;
mkdir "$SLURM_JOB_ID";
cp -R "$TMPDIR/models" "$SLURM_JOB_ID";
cp -R "$TMPDIR/experiments" "$SLURM_JOB_ID";
exit
"""
    with open(script_filename, "w") as file:
        file.write(script_text)


def write_start_script() -> None:
    start_script_text = f"""#!/bin/bash

for script in {JOB_FOLDER}/*main_loop*.sh
do
  echo "$script"
  sbatch "$script"
done
"""
    filename = f"{JOB_FOLDER}/start.sh"
    with open(filename, "w") as file:
        file.write(start_script_text)
    os.chmod(filename, 0o755)


def write_inference_script(
    name: str, checkpoint_hash: str, experiment_folder: str, script_filename: str,
) -> None:
    config_path = f"./experiments/{experiment_folder}/params.toml"

    script_text = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1 --ntasks-per-node=48
#SBATCH --account=PAS1769

set -x

# shellcheck disable=SC2064
trap "cd $SLURM_SUBMIT_DIR;mkdir $SLURM_JOB_ID;cp -R $TMPDIR/* $SLURM_JOB_ID;exit" TERM

module load python/3.7-2019.10

cd "$TMPDIR" || exit

python --version
python -m venv venv
# shellcheck disable=SC1091
. ./venv/bin/activate

pip install torch torchvision transformers==3.0.2 toml typing-extensions

cd "$SLURM_SUBMIT_DIR" || exit

CHECKPOINT=./models/{checkpoint_hash}.pt

cp aesw-test.csv "$TMPDIR"
cp -R paper "$TMPDIR"
cp -R experiments "$TMPDIR"
mkdir -p "$TMPDIR/models"
{make_copy_command(config_path)}
cp $CHECKPOINT "$TMPDIR/models"

cd "$TMPDIR" || exit

/usr/bin/time python -m paper {config_path} --preprocess test

/usr/bin/time python -m paper {config_path} --inference-test $CHECKPOINT --force-cpu

cd "$SLURM_SUBMIT_DIR" || exit;
mkdir "$SLURM_JOB_ID";
cp -R "$TMPDIR" "$SLURM_JOB_ID";
exit
"""

    with open(script_filename, "w") as file:
        file.write(script_text)


def main() -> None:
    for script in SCRIPTS:
        write_script(**script)

    for script in INFERENCE_SCRIPTS:
        write_inference_script(**script)

    write_start_script()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "job_folder", help="Where the job scripts will be written",
    )

    args = parser.parse_args()

    JOB_FOLDER = os.path.join(args.job_folder, "automated")

    shutil.rmtree(JOB_FOLDER, ignore_errors=True)
    os.makedirs(JOB_FOLDER)

    SCRIPTS: List[Dict[str, str]] = [
        # base with aesw
        {
            "name": "base_1e6",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_base_aesw_32_1e6",
            "script_filename": f"./{JOB_FOLDER}/aesw_main_loop_base_1e6.sh",
        },
        {
            "name": "base_2e5",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_base_aesw_32_2e5",
            "script_filename": f"./{JOB_FOLDER}/aesw_main_loop_base_2e5.sh",
        },
        {
            "name": "base_2e7",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_base_aesw_32_2e7",
            "script_filename": f"./{JOB_FOLDER}/aesw_main_loop_base_2e7.sh",
        },
        {
            "name": "base_1e4",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_base_aesw_32_1e4",
            "script_filename": f"./{JOB_FOLDER}/aesw_main_loop_base_1e4.sh",
        },
        # large with aesw
        {
            "name": "large_1e6",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_large_aesw_16_1e6",
            "script_filename": f"./{JOB_FOLDER}/aesw_main_loop_large_1e6.sh",
        },
        {
            "name": "large_2e5",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_large_aesw_16_2e5",
            "script_filename": f"./{JOB_FOLDER}/aesw_main_loop_large_2e5.sh",
        },
        {
            "name": "large_1e4",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_large_aesw_16_1e4",
            "script_filename": f"./{JOB_FOLDER}/aesw_main_loop_large_1e4.sh",
        },
        {
            "name": "large_2e7",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_large_aesw_16_2e7",
            "script_filename": f"./{JOB_FOLDER}/aesw_main_loop_large_2e7.sh",
        },
        # base with arxiv
        {
            "name": "arxiv_1e6",
            "copy_train_command": 'cp -R arxiv-train "$TMPDIR"',
            "experiment_folder": "bert_base_arxiv_32_1e6",
            "script_filename": f"./{JOB_FOLDER}/arxiv_main_loop_base_1e6.sh",
        },
        {
            "name": "arxiv_2e5",
            "copy_train_command": 'cp -R arxiv-train "$TMPDIR"',
            "experiment_folder": "bert_base_arxiv_32_2e5",
            "script_filename": f"./{JOB_FOLDER}/arxiv_main_loop_base_2e5.sh",
        },
        {
            "name": "arxiv_1e4",
            "copy_train_command": 'cp -R arxiv-train "$TMPDIR"',
            "experiment_folder": "bert_base_arxiv_32_1e4",
            "script_filename": f"./{JOB_FOLDER}/arxiv_main_loop_base_1e4.sh",
        },
        {
            "name": "arxiv_1e8",
            "copy_train_command": 'cp -R arxiv-train "$TMPDIR"',
            "experiment_folder": "bert_base_arxiv_32_1e8",
            "script_filename": f"./{JOB_FOLDER}/arxiv_main_loop_base_1e8.sh",
        },
        {
            "name": "arxiv_4e8",
            "copy_train_command": 'cp -R arxiv-train "$TMPDIR"',
            "experiment_folder": "bert_base_arxiv_32_4e8",
            "script_filename": f"./{JOB_FOLDER}/arxiv_main_loop_base_4e8.sh",
        },
        # base, pretrained on arxiv
        {
            "name": "pretrain_1e6",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_base_pretrain",
            "script_filename": f"./{JOB_FOLDER}/pretrain_main_loop_base_1e6.sh",
        },
        {
            "name": "pretrain_2e7",
            "copy_train_command": 'cp aesw-train.csv "$TMPDIR"',
            "experiment_folder": "bert_base_pretrain_2e7",
            "script_filename": f"./{JOB_FOLDER}/pretrain_main_loop_base_2e7.sh",
        },
    ]

    INFERENCE_SCRIPTS: List[Dict[str, str]] = [
        # base with aesw
        {
            "name": "base_1e6_inference",
            "checkpoint_hash": "c85f4d73-b35b-4298-8320-d58cbbbf1012",
            "experiment_folder": "bert_base_aesw_32_1e6",
            "script_filename": f"./{JOB_FOLDER}/aesw_inference_base_1e6.sh",
        },
        {
            "name": "base_2e5_inference",
            "checkpoint_hash": "dcf133c0-a269-4e1f-b41a-f5ca017a4492",
            "experiment_folder": "bert_base_aesw_32_2e5",
            "script_filename": f"./{JOB_FOLDER}/aesw_inference_base_2e5.sh",
        },
        # large with aesw
        {
            "name": "large_1e6_inference",
            "checkpoint_hash": "f65f93e7-3c8d-41ad-9e24-f71eab7b31d5",
            "experiment_folder": "bert_large_aesw_16_1e6",
            "script_filename": f"./{JOB_FOLDER}/aesw_inference_large_1e6.sh",
        },
        # base with arxiv
        {
            "name": "arxiv_1e6_inference",
            "checkpoint_hash": "bf04f1e6-10e3-4161-88a7-3ddd44ba6682",
            "experiment_folder": "bert_base_arxiv_32_1e6",
            "script_filename": f"./{JOB_FOLDER}/arxiv_inference_base_1e6.sh",
        },
        # pretrained base
        {
            "name": "pretrain_1e6_inference",
            "checkpoint_hash": "08eb79ff-152b-48b7-abdc-683efb35a27a",
            "experiment_folder": "bert_base_pretrain",
            "script_filename": f"./{JOB_FOLDER}/pretrain_inference_base_1e6.sh",
        },
    ]

    main()
