import collections
import re
import os.path
import argparse

def read_txt_file(file_name, include_con_info=False):
    """
    Take the text file data and output it with line number - sentence \t relationship number place holder
    :param file_name: record-x.txt
    :param include_con_info: if true, will append three more spots to the vector to indicate if problem, test, or
    treatments are present
    :return: a dictionary with line number as key and sentence, [0, 0, 0, 0, 0, 0, 0, 0/0, 0, 0] as value
    """
    # Hold the converted lines
    converted_lines = {}
    # variable to track line number to merge relationship and line
    line_num = 1

    # open the file parameter
    with open(file_name, 'r', encoding='utf-8') as f:

        # iterate through lines
        for file_line in f:

            # Remove newline character
            sentence = file_line.replace('\n', '')
            # Add line number, sentence, and default relationship value holder
            # Use a list for sentence and default relationship value
            if not include_con_info:
                converted_lines[line_num] = ("{:s}".format(sentence), [0, 0, 0, 0, 0, 0, 0, 0])
            else:
                converted_lines[line_num] = ("{:s}".format(sentence), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            line_num += 1

    return converted_lines


def get_sentences_concept_count(con_filename):
    """
    Take in a concept file and return a dictionary with a key representing txt file line number and the count of concepts\
    for that sentence
    :param con_filename: concept file
    :return: dictionary with key line number -> value count of concepts for that sentence
    """
    # Create dictionary
    concept_count = {}
    # Take in concept file
    with open(con_filename, 'r') as f:
        for file_line in f:
            # Get line number
            line_number = re.findall(r'\d+:', file_line)
            line_num = int( line_number[0].replace(':', '') )
            # Check dictionary has line number
            if concept_count.get(line_num) is not None:
                concept_count[line_num] += 1
            # Otherwise add it
            else:
                concept_count[line_num] = 1
    #print( sum(concept_count.values()) ) sanity check, this should match the number of lines in the concept file
    return concept_count


def filter_less_than_two_concepts(txt_dict, concept_count):
    """
     Create a dictionary containing the txt document sentences to only include sentences with multiple concepts
     ** Consider altering txt_dict instead to save memory **
    :param txt_dict: A dictionary containing sentence line numbers as keys
    :param concept_count: A dictionary containing a sentence line number as key with concept count of that sentence as value
    """
    multiconcept_txt_dict = {}
    # Iterate through txt_dict
    for line_num in txt_dict:
        # Check lineNum in concept file and concept count >= 2
        if concept_count.get(line_num) is not None and concept_count[line_num] >= 2:
            #print(line_num, ":", concept_count[line_num], "added")
            multiconcept_txt_dict[line_num] = txt_dict[line_num]
        # Otherwise, do nothing

    return multiconcept_txt_dict


def merge_files(txt_dict, rel_filename, con_filename=None):
    """
    Take in a dictionary formatted with linenum -> (sentence, [0, 0, 0, 0, 0, 0, 0, 0]) and merge with corresponding .rel file
    :param txt_dict: txt file converted to linenum -> (sentence, [0, 0, 0, 0, 0, 0, 0, 0]) format
    :param rel_filename: rel file corresponding to txt list
    :param con_filename: optional concept file corresponding to txt list, adds concept info to vector
    :return: output a file containing sentence \t relationship identifier number
    """

    # Take in rel file
    # Use regex to get line number
    # Can use regex to get relationship
    # Assign proper value indicating relationship
    # open the file parameter
    with open(rel_filename, 'r') as f:
        # iterate through lines
        for file_line in f:
            # Remove newline character
            sentence = file_line.replace('\n', '')
            # Get line number by looking for '\" num:' -> remove '\" 'and ':' to get line num
            line_number = re.findall(r'\" \d+:', file_line)
            line_num = int( line_number[0].replace(':', '').replace('\" ', '') )
            # Get relationship
            relationship = re.search(r'(r=)(["])(\w{3}|\w{4}|\w{5})(["])', file_line)
            rel = relationship.group(0).replace('r=', '')

            # Update relationship vector
            if rel == '"TrIP"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 0)

            elif rel == '"TrWP"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 1)

            elif rel == '"TrCP"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 2)

            elif rel == '"TrAP"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 3)

            elif rel == '"TrNAP"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 4)

            elif rel == '"TeRP"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 5)

            elif rel == '"TeCP"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 6)

            elif rel == '"PIP"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 7)

            else:
                print("Invalid rel: ", rel)

    # If concept file provided
    # Take in con file
    # Use regex to get line number
    # Can use regex to get concept type
    # Assign proper value indicating concept
    #   index 8 = treatment, index 9 = test, index 10 = problem

    if not con_filename:
        return None
    # open the file parameter
    with open(con_filename, 'r') as f:
        # iterate through lines
        for file_line in f:
            # Remove newline character
            sentence = file_line.replace('\n', '')
            # Get line number by looking for '\" num:' -> remove '\" 'and ':' to get line num
            line_number = re.findall(r'\" \d+:', file_line)
            line_num = int( line_number[0].replace(':', '').replace('\" ', '') )
            # Get concept
            concept = re.search(r'(t=)(["])(\w{4}|\w{7}|\w{9})(["])', file_line)
            con = concept.group(0).replace('t=', '')

            # Ensure line number was not filtered out
            if txt_dict.get(line_num) is None:
                continue

            # Update concept vector
            if con == '"treatment"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 8)
            elif con == '"test"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 9)
            elif con == '"problem"':
                txt_dict[line_num] = update_merge_vec(txt_dict[line_num], 10)
            else:
                print("Invalid con: ", con)


def update_merge_vec(tupl, index):
    """
    :param tupl: tuple containing sentence and relationship/con vector
    :param index: vector index to update
    :return: tuple with sentence and updated relationship vector
    """
    # Get sentence from tuple
    sentence = tupl[0]
    # Update individual index in the relationship vector
    vector = tupl[1]
    vector[index] = 1
    # output new tuple for dictionary
    return (sentence, vector)


def create_file(out_file_path, header_list):
    """
    Create output file with proper headers that write_file function will append
    :param out_file_path: Path and name of the output file
    :param header_list: Headers of the file
    :return: output file path
    """
    # Output file path and name
    output_file = out_file_path

    delim = "\t"

    # Create tsv file with headers
    with open(output_file, 'w+') as out:
        # Add header
        header_line = delim.join(header_list)
        out.write(header_line + "\n")

    # Return file path
    return output_file


def write_file(file_dict, out_file_path):
    """
    Append sentence and corresponding relatinship/concept vector to its own tsv file line
    :param file_dict: dictionary of txt rel/con file pair
    :param out_file_path: path of the file to be appended
    :return: number of lines with no relationships and with relationships
    """

    delim = "\t"

    # Initialize variables
    no_relation = 0
    relation = 0

    with open(out_file_path, 'a+') as out:

        # Go through dictionary
        for key in file_dict:
            # Get the tuple containing sentence and dictionary
            tup = file_dict[key]
            # Get tuple values
            sentence = tup[0]
            vector = tup[1]
            # Lopressor 50 " -> problem sentence for csv viewer

            # Count how many lines don't have a relation
            #if collections.Counter(vector) == collections.Counter([0, 0, 0, 0, 0, 0, 0, 0]):
            if sum(vector[0:8:1]) == 0:
                no_relation += 1
            else:
                relation += 1

            # Create file line sentence\trel\trel\trel....
            # If last file line, remove end newline character
            final_tup = list(file_dict.items())[-1]
            if tup == final_tup:
                # Convert all elements to string
                temp = list(map(str, vector))
                # Make relationship vector a string separating each relationship with \t
                rel_string = delim.join(temp)
                file_line = "{}\t{}".format(sentence, rel_string)
            else:
                # Convert all elements to string
                temp = list(map(str, vector))
                # Make relationship vector a string separating each relationship with \t
                rel_string = delim.join(temp)
                file_line = '{}\t{}\n'.format(sentence, rel_string)
                #file_line = sentence + delim + rel_string + "\n"
            # Write to file
            out.write(file_line)

    return (no_relation, relation)


def append_final_file(output_file_path, directory_path, filter_concepts=False, include_con_info=False):
    """
    Loop through rel directory, marry rel and txt files to dictionary, append that dictionary to the output file
    :param output_file_path: file path of final tsv to contain all data
    :param directory_path: path to directory containing rel files
    :param filter_concepts: boolean indicating whether or not to filter out sentences with less than two concepts
    :param include_con_info: option to include three additional indexes representing concept types present in the sentence
    """
    # Loop through directory
    total_no_relation = 0
    total_relation = 0
    for rel_filename in os.listdir(directory_path):
        # Extract corresponding file names
        rel_file_path = directory_path + "/" + rel_filename
        txt_file_path = rel_file_path.replace("/rel", "/txt").replace(".rel", ".txt")
        concept_file_path = rel_file_path.replace(".rel", ".con").replace("/rel", "/concept")
        # Get linenum dictionary
        txt_dict = read_txt_file(txt_file_path, include_con_info=include_con_info)

        # Filter out sentences with less than two concepts (if desired)
        if filter_concepts:
            # Get concept count dictionary
            concept_count = get_sentences_concept_count(concept_file_path)
            # Filter and overwrite the txt_dict
            txt_dict = filter_less_than_two_concepts(txt_dict, concept_count)
           
        # Merge files
        if include_con_info:
            merge_files(txt_dict, rel_file_path, con_filename=concept_file_path)
        else:
            merge_files(txt_dict, rel_file_path)
        # Append merged dictionary to output file
        write_file_tuple = write_file(txt_dict, output_file_path)
        total_no_relation += write_file_tuple[0]
        total_relation += write_file_tuple[1]
    # Print total lines with no relation
    print("No relation:", total_no_relation, "Relation:", total_relation)


if __name__ == '__main__':

    # run script with a command such as:
    #   python3 convert_2010i2b2_relex.py concept_assertion_relation_training_data/beth/rel concept_assertion_relation_training_data/partners/rel test_data/rel training.tsv test.tsv training_and_test.tsv 0 0

    # grab arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("beth_rel_dir", help="The path where the .rel files for Beth Israel are stored.")
    parser.add_argument("partners_rel_dir", help="The path where the .rel files for Partners are stored.")
    parser.add_argument("test_rel_dir", help="The path where the .rel files for the test dataset are stored")
    parser.add_argument("training_output_path", help="The path where the converted training data is output")
    parser.add_argument("test_output_path", help="The path where the converted test data is output")
    parser.add_argument("training_and_test_output_path", help="The path where the converted training+test data is output")
    parser.add_argument("filter_less_than_two_concepts", help="0 or 1, 1 indicates that sentences with less than 2 concepts will be removed")
    parser.add_argument("include_concept_info", help="0 or 1, 1 indicates that a vector indicating which concepts are present will be included")
    args = parser.parse_args()
    # convert from int arguments to boolean
    args.filter_less_than_two_concepts = int(args.filter_less_than_two_concepts) > 0
    args.include_concept_info = int(args.include_concept_info) > 0

    # hard-coded header lists for output files
    header_list_with_concept_info = ['Sentence', 'TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'TeRP', 'TeCP', 'PIP', 'CTr', 'CTe', 'CPr']
    reg_header_list = ['Sentence', 'TrIP', 'TrWP', 'TrCP', 'TrAP', 'TrNAP', 'TeRP', 'TeCP', 'PIP']
    header_list = reg_header_list
    if int(args.include_concept_info) > 0:
        header_list = header_list_with_concept_info

    # create the training file
    training_output_file = create_file(args.training_output_path, header_list)
    append_final_file(output_file_path=args.training_output_path, directory_path=args.beth_rel_dir,
                      filter_concepts=args.filter_less_than_two_concepts, include_con_info=args.include_concept_info)
    append_final_file(output_file_path=args.training_output_path, directory_path=args.partners_rel_dir,
                      filter_concepts=args.filter_less_than_two_concepts, include_con_info=args.include_concept_info)

    # create the test file
    test_output_file = create_file(args.test_output_path, header_list)
    append_final_file(output_file_path=args.test_output_path, directory_path=args.test_rel_dir,
                      filter_concepts=args.filter_less_than_two_concepts, include_con_info=args.include_concept_info)

    # create the combined file
    training_and_test_output_file = create_file(args.training_and_test_output_path, header_list)
    append_final_file(output_file_path=args.training_and_test_output_path, directory_path=args.beth_rel_dir,
                      filter_concepts=args.filter_less_than_two_concepts, include_con_info=args.include_concept_info)
    append_final_file(output_file_path=args.training_and_test_output_path, directory_path=args.partners_rel_dir,
                      filter_concepts=args.filter_less_than_two_concepts, include_con_info=args.include_concept_info)
    append_final_file(output_file_path=args.training_and_test_output_path, directory_path=args.test_rel_dir,
                      filter_concepts=args.filter_less_than_two_concepts, include_con_info=args.include_concept_info)