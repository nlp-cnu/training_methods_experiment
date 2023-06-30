import os
from random import shuffle
from ast import literal_eval

def combine_for_nlmchem():
    # collect all of the positive chemical/drug/medication samples from
    # CDR, BC7DCPI, BC7MED, i2b2, n2c2
    cdr_dataset = os.path.join("..", "..", "cdr", "converted_all.tsv")
    bc7med_dataset = os.path.join("..", "..", "bc7med", "converted_all.tsv")
    bc7dcpi_dataset = os.path.join("..", "..", "bc7dcpi", "converted_all.tsv")
    i2b2_dataset = os.path.join("..", "..", "i2b2", "converted_all.tsv")
    n2c2_dataset = os.path.join("..", "..", "n2c2", "converted_all.tsv")

    intermediate_datasets = [cdr_dataset, bc7med_dataset, bc7dcpi_dataset, i2b2_dataset, n2c2_dataset]
    relevant_classes = [[1], [1], [1], [2], [1]]

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
    combine_for_nlmchem()
