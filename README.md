# Rubric-Assisted Automated Short Answer Grading

This repository contains code for the paper: [Enhancing Explainability and Performance in Automated Short Answer Grading through Rubric-Based Assessment](https://arxiv.org/abs/xxxx.yyyyy)

## Dependencies
Please install dependencies in [requirements.txt](https://github.com/luffycodes/Rubric-ASAG/blob/main/requirements.txt).

## Preprocessing
[dataset/preprocess.py](https://github.com/luffycodes/Rubric-ASAG/blob/main/dataset/preprocess.py) preprocesses the raw data into correct formats.

Usage:
```
python dataset/preprocess.py --dataset_name CHEM121 --data_dir data --train_size 0.8 --valid_size 0.1 --test_size 0.1 --seed 42
```

## Running encoder models
[encoder_model/train_and_evaluate.py](https://github.com/luffycodes/Rubric-ASAG/blob/main/encoder_model/train_and_evaluate.py) trains and evaluates encoder models on the processed data.

Usage:
```
# Running on data with rubrics
python encoder_model/train_and_evaluate.py --dataset_name CHEM121 --data_dir data --train_size 0.8 --valid_size 0.1 --test_size 0.1 --seed 42 --use_rubric --metric_for_best_model f1 --model roberta-base --num_train_epochs 10 --train_batch_size 16 --eval_batch_size 16 --learning_rate 0.00002 --cache_dir cache --output_dir output

# Running on data without rubrics
python encoder_model/train_and_evaluate.py --dataset_name CHEM121 --data_dir data --train_size 0.8 --valid_size 0.1 --test_size 0.1 --seed 42 --metric_for_best_model f1 --model roberta-base --num_train_epochs 10 --train_batch_size 16 --eval_batch_size 16 --learning_rate 0.00002 --cache_dir cache --output_dir output
```

## Running decoder models
[decoder_model/prompt_gpt.py](https://github.com/luffycodes/Rubric-ASAG/blob/main/decoder_model/prompt_gpt.py) prompts GPT with the preprocessed data and stores results using IO redirect.

Usage:
```
# Running a new experiment
python decoder_model/prompt_gpt.py --dataset_name CHEM121 --data_dir data --train_size 0.8 --valid_size 0.1 --test_size 0.1 --seed 42 --use_rubric --engine gpt-4 --prompt_choice zero_shot --max_tokens 2048 --output_dir output

# Resuming the experiment
python decoder_model/prompt_gpt.py --dataset_name CHEM121 --data_dir data --train_size 0.8 --valid_size 0.1 --test_size 0.1 --seed 42 --use_rubric --engine gpt-4 --prompt_choice zero_shot --max_tokens 2048 --output_dir output
```

[decoder_model/evaluate_gpt.py](https://github.com/luffycodes/Rubric-ASAG/blob/main/decoder_model/evaluate_gpt.py) evalutes the GPT IO output text file.

Usage:
```
# Evaluating results on data with rubrics
python decoder_model/evaluate_gpt.py --gpt_io_file some_gpt_io_file.txt --use_rubric

# Evaluating results on data without rubrics
python decoder_model/evaluate_gpt.py --gpt_io_file some_gpt_io_file.txt
```

If you use this work, please cite:
Enhancing Explainability and Performance in Automated Short Answer Grading through Rubric-Based Assessment
(https://arxiv.org/abs/xxxx.yyyyy)
```
@misc{kangqi2023rasag,
      title={Enhancing Explainability and Performance in Automated Short Answer Grading through Rubric-Based Assessment}, 
      author={Kangqi Ni and Shashank Sonkar and Richard G. Baraniuk},
      year={2023},
      eprint={xxxx.yyyyy},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
