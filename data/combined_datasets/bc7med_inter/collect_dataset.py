import os
from random import shuffle
from ast import literal_eval

def combine_for_bc7med():
    # collect all of the positive drug samples from
    # n2c2
    n2c2_dataset = os.path.join("..", "..", "n2c2", "converted_all.tsv")

    intermediate_datasets = [n2c2_dataset]
    relevant_classes = [[1]]

    output_file = os.path.join("converted_all.tsv")
    positive_label = 1
    negative_label = 0

    all_lines = []
    with open(output_file, 'w+') as of:
        for dataset, relevant_classes in zip(intermediate_datasets, relevant_classes):
            with open(dataset, 'r+') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                text, annotation = line.split("\t")
                annotation = literal_eval(annotation)
                include = False
                new_annotation = []
                for tag in annotation:
                    if tag in relevant_classes:
                        new_annotation.append(positive_label)
                        include = True
                    else:
                        new_annotation.append(negative_label)

                if include:
                    assert(len(new_annotation) == len(annotation))
                    of.write(f"{text}\t{new_annotation}\n")


    with open(output_file, 'r+') as of:
        lines = of.readlines()

    shuffle(lines)

    with open(output_file, 'w+') as of:
        lines = of.writelines(lines)


if __name__ == "__main__":
    combine_for_bc7med()
