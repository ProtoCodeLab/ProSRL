import datetime
import os.path

from ParsingSource import *
from Tools import *
#import tool.javalang as jl
import javalang as jl

# Start Time
start_time = datetime.datetime.now()
start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

def parse_source(project_root_path, handcraft_file_names, package_heads):
    result = {}
    count = 0
    existed_file_names = []
    for dir_path, dir_names, file_names in os.walk(project_root_path):

        if len(file_names) == 0:
            continue

        index = -1
        for _head in package_heads:
            index = int(dir_path.find(_head))
            if index >= 0:
                break
        if index < 0:
            continue

        package_name = dir_path[index:]
        package_name = package_name.replace(os.sep, '.')

        for file in file_names:
            if file.endswith('java'):
                if str(package_name + "." + str(file)) not in handcraft_file_names:
                    continue

                ast_result  = ast_parse_CPDP(str(os.path.join(dir_path, file)))

                if ast_result:  # ast_result != []
                    result[package_name + "." + str(file)] = ast_result
                    existed_file_names.append(str(package_name + "." + str(file)))
                    count += 1

    for handcraft_file_name in handcraft_file_names:
        handcraft_file_name.replace('.java', '')
        if handcraft_file_name not in existed_file_names:
            print('This file is Parsing failed:' + handcraft_file_name)

    print("data size : " + str(count))
    # ratio = len(existed_file_names) / len(handcraft_file_names)
    # print(f"Proportion: {ratio:.1%}")
    return result

def ast_parse_CPDP(source_file_path):
    with open(source_file_path, 'rb') as file_obj:
        content = file_obj.read()
        result = set()
        tree = []
        try:
            tree = jl.parse.parse(content)
        except jl.parser.JavaSyntaxError as e:
            print('JavaSyntaxError:')
            print(source_file_path)
            print(e.description)
            print(e.at)
        excluded_types = {"CompilationUnit"}
        for path, node in tree:
            node_type = type(node).__name__
            if node_type not in excluded_types:
                result.add(node_type)
                print(f"Node Type: {node_type}")
                print('-----------')
        print(result)
        return result

def main():
    REGENERATE = False
    dump_data_path = '../data_loader/dic/'#'/root/autodl-tmp/'
    if not os.path.exists(dump_data_path):
        os.makedirs(dump_data_path)

    root_path_source = '../data/projects/'
    root_path_csv = '../data/csvs/'
    package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Analyze projects
    path_projects = []
    with open('../data/pairs-pre1.txt', 'r') as file_obj:
        for line in file_obj:
            stripped_line = line.strip()
            if stripped_line:
                path_projects.append(stripped_line)

    path_set = os.path.join(dump_data_path, 'global_vocabulary.pkl')

    if os.path.exists(path_set) and not REGENERATE:
        print("Loading saved data")
        obj = load_data(path_set)
        [vector_len, vocabulary] = obj
        # print("vocabulary (type to index mapping):")
        # print(vocabulary)
    else:
        global_node_types = set()
        for project in path_projects:
            path_source = root_path_source + project
            path_handcraft = root_path_csv + project + '.csv'

            # If you don't need to regenerate, get it directly from dump_data

            file_instances = extract_handcraft_instances(path_handcraft)

            # Get tokens
            file_node_types = parse_source(path_source, file_instances, package_heads)

            for node_types in file_node_types.values():
                global_node_types.update(node_types)

        # Generating a global dictionary
        global_vocabulary = {typ: idx + 1 for idx, typ in enumerate(sorted(global_node_types))}
        print(global_vocabulary)
        global_vector_len = len(global_vocabulary)
        # Saving a global dictionary
        dump_data(os.path.join(dump_data_path, 'global_vocabulary.pkl'), [global_vector_len, global_vocabulary])

if __name__ == "__main__":
    main()
