# Automated Long Answer Grading with RiceChem Dataset

This repository contains code for the paper: [Automated Long Answer Grading with RiceChem Dataset](https://arxiv.org/abs/2404.14316)

## Request for Data
Please fill out this [google form](https://forms.gle/d3sYD5vMXnK5aMKo6) to request the RiceChem dataset. Place the data folder into the root dir.

## Dependencies
Install dependencies with [requirements.txt](https://github.com/luffycodes/Automated-Long-Answer-Grading/blob/review/requirements.txt).

## Preprocess
[preprocess/preprocess.py](https://github.com/luffycodes/Automated-Long-Answer-Grading/blob/review/preprocess/preprocess.py) preprocesses raw data in the data dir into required formats.
```
# Preprocess the raw data of all questions uniformly into splits
python preprocess/preprocess.py --data_dir data --output_dir data_processed --train_size 0.8 --valid_size 0.1 --test_size 0.1
```
```
# Preprocess the raw data while using some questions as completely unseen test splits
python preprocess/preprocess.py --data_dir data --output_dir data_processed --train_size 0.8 --valid_size 0.2 --test_size 1.0 --unseen_split
```

## Run encoder models
[encoder_model/run_encoder_models.py](https://github.com/luffycodes/Automated-Long-Answer-Grading/blob/review/encoder_model/run_encoder_model.py) trains and evaluates encoder models on the processed data.
```
# Run with rubrics (Section 4.2 Benchmarking on Discriminative Models + Section 4.3 The Value of Entailment Formulation in ALAG)
CUDA_VISIBLE_DEVICES=0 python encoder_model/run_encoder_model.py --data_dir data_processed --train_size 0.8 --valid_size 0.1 --test_size 0.1 --seed 42 --use_rubric --metric_for_best_model f1 --model roberta-large-mnli --num_train_epochs 10 --train_batch_size 16 --eval_batch_size 16 --learning_rate 0.00002
```
```
# Run without rubrics (Section 4.4 The Importance of Rubric-based Formulation in ALAG)
CUDA_VISIBLE_DEVICES=0 python encoder_model/run_encoder_model.py --data_dir data_processed --train_size 0.8 --valid_size 0.1 --test_size 0.1 --seed 42 --metric_for_best_model f1 --model roberta-large-mnli --num_train_epochs 10 --train_batch_size 16 --eval_batch_size 16 --learning_rate 0.00002
```
```
# Run with rubrics and use question 0 as unseen test data (Section 4.5 Benchmarking on Cold Start Scenarios)
CUDA_VISIBLE_DEVICES=0 python encoder_model/run_encoder_model.py --data_dir data_processed --train_size 0.8 --valid_size 0.2 --test_size 1.0 --seed 42 --use_rubric --unseen_split 0 --metric_for_best_model f1 --model roberta-large-mnli --num_train_epochs 10 --train_batch_size 16 --eval_batch_size 16 --learning_rate 0.00002
```

## Run decoder models
[decoder_model/run_decoder_models.py](https://github.com/luffycodes/Automated-Long-Answer-Grading/blob/review/decoder_model/run_decoder_model.py) evaluates encoder models(LLMs) on the processed data.
```
# Run GPT (Section 4.6 Benchmarking on Large Language Models)
export OPENAI_API_KEY="key"
python decoder_model/run_decoder_model.py --test_size 0.1 --seed 42 --use_rubric --model gpt-3.5-turbo-0125

# Run open-sourced LLMs (Section 4.6 Benchmarking on Large Language Models)
CUDA_VISIBLE_DEVICES=0 python decoder_model/run_decoder_model.py --test_size 0.1 --seed 42 --use_rubric --model mistralai/Mistral-7B-Instruct-v0.2
```


<br> If you use this work, please cite:
```
@misc{sonkar2024automated,
      title={Automated Long Answer Grading with RiceChem Dataset}, 
      author={Shashank Sonkar and Kangqi Ni and Lesa Tran Lu and Kristi Kincaid and John S. Hutchinson and Richard G. Baraniuk},
      year={2024},
      eprint={2404.14316},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
