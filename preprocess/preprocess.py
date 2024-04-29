import csv
import random
import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="CHEM121", help="the name of the dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="the directory to the data")
    parser.add_argument("--train_size", type=float, required=True, help="the size of the training dataset")
    parser.add_argument("--valid_size", type=float, required=True, help="the size of the valid dataset")
    parser.add_argument("--test_size", type=float, required=True, help="the size of the testing dataset")
    parser.add_argument("--seed", type=int, default=42, help="the seed for the random module")
    parser.add_argument("--output_dir", type=str, required=True, help="the processed data dir")
    parser.add_argument("--unseen_split", action='store_true', help="whether to split data for unseen questions")
    
    params = parser.parse_args()
    return params

def concat_path(data_dir, answer_files, rubric_files):
    prefix = data_dir + '/'
    for i in range(len(answer_files)):
        answer_files[i] = prefix + answer_files[i]
    for i in range(len(rubric_files)):
        rubric_files[i] = prefix + rubric_files[i]

def map_ids_answers(answer_file):
    id2ans = dict()
    with open(answer_file, 'r') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader) 

        # Map the id to the answer
        for row in reader:
            id = row[0]
            ans = row[1]

            # Skip the blank answers
            if ans == '':
                continue

            id2ans[id] = ans
    return id2ans

def get_questions(answer_csvs):
    id2ques = dict()

    for i, answer_csv in enumerate(answer_csvs):
        with open(answer_csv, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)[1:]

            question = header[0]
            question = question[question.find(':')+1:].strip()
            rubric = header[1]

            rubric_items = []
            for item in rubric.split('\n'):
                item = item.strip()
                if item in invalid_items:
                    continue
                rubric_items.append(item)

        id2ques[i] = {"question": question, "items": rubric_items}
            
    return id2ques

def add_data(index, answer_files, rubric_files, dataset_csvs_rubric, dataset_csvs_no_rubric, splitted_dataset_csvs, dataset_args, invalid_items):
    answer_file = answer_files[index]
    rubric_file = rubric_files[index]
    
    # Map student IDs to answers
    id2ans = map_ids_answers(answer_file)

    # Format the datasets
    answers = [[], [], []]
    rubrics = [[], [], []]
    scores = [[], [], []]

    with open(rubric_file, 'r') as file:
        reader = csv.reader(file)

        # Get all the rubric items
        header = next(reader)[1:]

        # Get all the data
        rows = list(reader)
        random.seed(dataset_args.seed)
        random.shuffle(rows)

        # Split the data
        total_rows = len(rows)
        num_train = int(dataset_args.train_size * total_rows)
        num_valid = int(dataset_args.valid_size * total_rows)
        splitted_rows = [rows[:num_train], rows[num_train: num_train + num_valid], rows[num_train + num_valid:]]

        # Split the raw data into separate csvs    
        for i in range(3):
            with open(splitted_dataset_csvs[i], 'a') as file:
                writer = csv.writer(file)
                for row in splitted_rows[i]:
                    id = row[0]
                    
                    # Skip if no corresponding answers
                    if id not in id2ans:
                        continue

                    ans = id2ans[id]
                    values = []
                    for item, value in zip(header, row[1:]):
                        if item.lower() in invalid_items:
                            continue
                        values.append(value)
                    
                    writer.writerow([index, ans] + values + [len(values)])

        # Fill in train, valid, and test datasets
        for i in range(3):
            for row in splitted_rows[i]:
                # Get the student id
                id = row[0]

                # Skip if no corresponding answers
                if id not in id2ans:
                    continue
                
                total = 0
                acc = 0
                item2value = dict()
                for item, value in zip(header, row[1:]):
                    # Skip invalid rubric items
                    if item.lower() in invalid_items:
                        continue
                    
                    item2value[item] = 1 if value == 'TRUE' else 0

                    if value == 'TRUE':
                        acc += 1
                    total += 1

                answers[i].append(id2ans[id]) # the student answer
                rubrics[i].append(item2value) # the rubric items

                score = int(acc / total * 8)
                scores[i].append(score) # the score

    for i in range(3):
        # Add data to the datasets with rubrics
        with open(dataset_csvs_rubric[i], 'a') as file:
            writer = csv.writer(file)
            for j in range(len(answers[i])):
                answer = answers[i][j]
                rubric = rubrics[i][j]
                for item in rubric:
                    # Write in the premise, hypothesis, and label
                    writer.writerow([answer, item, rubric[item]])

        # Add data to the datasets without rubrics
        with open(dataset_csvs_no_rubric[i], 'a') as file:
            writer = csv.writer(file)
            for j in range(len(answers[i])):
                # Write in the sentence and score
                writer.writerow([answers[i][j], scores[i][j]])

def write_headers(dataset_csvs_rubric, dataset_csvs_no_rubric, splitted_dataset_csvs):
    for i in range(3):
        with open(dataset_csvs_rubric[i], 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['premise', 'hypothesis', 'label'])

        with open(dataset_csvs_no_rubric[i], 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['sentence', 'label'])

        open(splitted_dataset_csvs[i], 'w').close()

def print_sizes(csv_files):
    for i in range(len(csv_files)):
        with open(csv_files[i], 'r') as file:
            reader = csv.reader(file)
            print(f'File: {csv_files[i]} Size: {len(list(reader)) - 1}')

def split_datasets(dataset_args, answer_files, rubric_files, invalid_items):
    # Create the dataset file names
    sizes = str(dataset_args.train_size) + '_' + str(dataset_args.valid_size) + '_' + str(dataset_args.test_size)

    file_prefix = dataset_args.output_dir + '/' + dataset_args.dataset_name + "_rubric_" + sizes
    os.makedirs(file_prefix, exist_ok=True)
    dataset_csvs_rubric = [file_prefix + "/train.csv", file_prefix + "/valid.csv", file_prefix + "/test.csv"]
    
    file_prefix = dataset_args.output_dir + '/' + dataset_args.dataset_name + "_no-rubric_" + sizes
    os.makedirs(file_prefix, exist_ok=True)
    dataset_csvs_no_rubric = [file_prefix + "/train.csv", file_prefix + "/valid.csv", file_prefix + "/test.csv"]

    file_prefix = dataset_args.output_dir + '/' + dataset_args.dataset_name + "_" + sizes
    os.makedirs(file_prefix, exist_ok=True)
    splitted_dataset_csvs = [file_prefix + '/train.csv',
                             file_prefix + '/valid.csv',
                             file_prefix + '/test.csv']
    
    # Write the csv headers
    write_headers(dataset_csvs_rubric, dataset_csvs_no_rubric, splitted_dataset_csvs)

    # Add the data into the training, validation, testing datasets
    for i in range(len(answer_files)):
        add_data(i, answer_files, rubric_files, dataset_csvs_rubric, dataset_csvs_no_rubric, splitted_dataset_csvs, dataset_args, invalid_items)

    # Verify the sizes of the datasets with rubrics
    print_sizes(dataset_csvs_rubric)

    # Verify the sizes of the datasets without rubrics
    print_sizes(dataset_csvs_no_rubric)

def add_data_unseen(answer_files, rubric_files, dataset_csvs_rubric, dataset_csvs_no_rubric, splitted_dataset_csvs, dataset_args, invalid_items, test_question_idx):
    for index in range(len(answer_files)):
        answer_file = answer_files[index]
        rubric_file = rubric_files[index]
        
        # Map student IDs to answers
        id2ans = map_ids_answers(answer_file)
    
        # Format the datasets
        answers = [[], [], []]
        rubrics = [[], [], []]
        scores = [[], [], []]
    
        with open(rubric_file, 'r') as file:
            reader = csv.reader(file)
    
            # Get all the rubric items
            header = next(reader)[1:]
    
            # Get all the data
            rows = list(reader)
            random.seed(dataset_args.seed)
            random.shuffle(rows)

            # Split the data
            total_rows = len(rows)
            if index != test_question_idx:
                num_train = int(dataset_args.train_size * total_rows)
                num_valid = int(dataset_args.valid_size * total_rows)
                splitted_rows = [rows[:num_train], rows[num_train:num_train+num_valid], []]
            else:
                num_test = int(dataset_args.test_size * total_rows)
                splitted_rows = [[], [], rows[:num_test]]
        
            # Split the raw data into separate csvs    
            for i in range(3):
                with open(splitted_dataset_csvs[i], 'a') as file:
                    writer = csv.writer(file)
                    for row in splitted_rows[i]:
                        id = row[0]
                        
                        # Skip if no corresponding answers
                        if id not in id2ans:
                            continue
    
                        ans = id2ans[id]
                        values = []
                        for item, value in zip(header, row[1:]):
                            if item.lower() in invalid_items:
                                continue
                            values.append(value)
                        
                        writer.writerow([index, ans] + values + [len(values)])
    
            # Fill in train, valid, and test datasets
            for i in range(3):
                for row in splitted_rows[i]:
                    # Get the student id
                    id = row[0]
    
                    # Skip if no corresponding answers
                    if id not in id2ans:
                        continue
                    
                    total = 0
                    acc = 0
                    item2value = dict()
                    for item, value in zip(header, row[1:]):
                        # Skip invalid rubric items
                        if item.lower() in invalid_items:
                            continue
                        
                        item2value[item] = 1 if value == 'TRUE' else 0
    
                        if value == 'TRUE':
                            acc += 1
                        total += 1
    
                    answers[i].append(id2ans[id]) # the student answer
                    rubrics[i].append(item2value) # the rubric items
    
                    score = int(acc / total * 8)
                    scores[i].append(score) # the score
    
        for i in range(3):
            # Add data to the datasets with rubrics
            with open(dataset_csvs_rubric[i], 'a') as file:
                writer = csv.writer(file)
                for j in range(len(answers[i])):
                    answer = answers[i][j]
                    rubric = rubrics[i][j]
                    for item in rubric:
                        # Write in the premise, hypothesis, and label
                        writer.writerow([answer, item, rubric[item]])
    
            # Add data to the datasets without rubrics
            with open(dataset_csvs_no_rubric[i], 'a') as file:
                writer = csv.writer(file)
                for j in range(len(answers[i])):
                    # Write in the sentence and score
                    writer.writerow([answers[i][j], scores[i][j]])

def split_datasets_unseen(dataset_args, answer_files, rubric_files, invalid_items, test_question_idx):
    # Create the dataset file names
    sizes = str(dataset_args.train_size) + '_' + str(dataset_args.valid_size) + '_' + str(dataset_args.test_size)

    file_prefix = dataset_args.output_dir + '/' + dataset_args.dataset_name + "_rubric_" + sizes + f"_unseen-{test_question_idx}"
    os.makedirs(file_prefix, exist_ok=True)
    dataset_csvs_rubric = [file_prefix + "/train.csv", file_prefix + "/valid.csv", file_prefix + "/test.csv"]

    file_prefix = dataset_args.output_dir + '/' + dataset_args.dataset_name + "_no-rubric_" + sizes + f"_unseen-{test_question_idx}"
    os.makedirs(file_prefix, exist_ok=True)
    dataset_csvs_no_rubric = [file_prefix + "/train.csv", file_prefix + "/valid.csv", file_prefix + "/test.csv"]

    file_prefix = dataset_args.output_dir + '/' + dataset_args.dataset_name + "_" + sizes + f"_unseen-{test_question_idx}"
    os.makedirs(file_prefix, exist_ok=True)
    splitted_dataset_csvs = [file_prefix + '/train.csv',
                             file_prefix + '/valid.csv',
                             file_prefix + '/test.csv']

    # Write the csv headers
    write_headers(dataset_csvs_rubric, dataset_csvs_no_rubric, splitted_dataset_csvs)

    # Add the data into the training, validation, testing datasets
    add_data_unseen(answer_files, rubric_files, dataset_csvs_rubric, dataset_csvs_no_rubric, splitted_dataset_csvs, dataset_args, invalid_items, test_question_idx)

    # Verify the sizes of the datasets with rubrics
    print_sizes(dataset_csvs_rubric)

    # Verify the sizes of the datasets without rubrics
    print_sizes(dataset_csvs_no_rubric)

if __name__ == "__main__":
    # Read user arguments
    args = parse_args()

    # List files
    answer_files = ["Sample CHEM121 Student Answers.xlsx - F20 Exam 2 Q4.csv",
                "Sample CHEM121 Student Answers.xlsx - F20 Exam 2 Q7.csv",
                "Sample CHEM121 Student Answers.xlsx - F20 Exam 3 Q7.csv",
                "Sample CHEM121 Student Answers.xlsx - F20 Final Exam Q3.csv"
                ]
    rubric_files = ["Sample CHEM121 Gradescope Export.xlsx - F20 Exam 2 Q4.csv",
                "Sample CHEM121 Gradescope Export.xlsx - F20 Exam 2 Q7.csv",
                "Sample CHEM121 Gradescope Export.xlsx - F20 Exam 3 Q7.csv",
                "Sample CHEM121 Gradescope Export.xlsx - F20 Final Exam Q3.csv"
                ]

    # Set the invalid rubric items
    invalid_items = set(['score', 'adjustment', 'comments', 'blank', 'incorrect', 'core charge calculation error',
                            'incorrect statement included', 'correct response', 'incorrect/blank response', 'incorrect/missing answer', 'incorrect/misleading statement'])

    # Concat the paths with the data dir
    concat_path(args.data_dir, answer_files, rubric_files)
    
    # Split the datasets
    if args.unseen_split:
        for i in range(4):
            split_datasets_unseen(args, answer_files, rubric_files, invalid_items, i)
    else:
        split_datasets(args, answer_files, rubric_files, invalid_items)

    # Map ids to questions and rubrics
    id2ques = get_questions(answer_files)
    with open(args.output_dir + "/question_rubrics.json", 'w') as file:
        json.dump(id2ques, file)