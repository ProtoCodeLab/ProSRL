import datetime
import itertools
from imblearn.over_sampling import RandomOverSampler
from ParsingSource import *
from Tools import *
#from tool import javalang
import javalang
import javalang.tree as jlt
import javalang as jl
import pandas as pd
from scipy.stats import fisher_exact
#from statsmodels.stats.multitest import multipletests
import random
import numpy as np
from collections import Counter

# Start Time
start_time = datetime.datetime.now()
start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

def random_oversample(file_node_types, labels):

    assert len(file_node_types) == len(labels)

    keys = list(file_node_types.keys())
    features = list(file_node_types.values())
    labels = np.array(labels)


    class_counts = Counter(labels)
    max_count = max(class_counts.values())

    new_keys = keys.copy()
    new_features = features.copy()
    new_labels = labels.tolist()

    key_counts = Counter()

    for class_label, count in class_counts.items():
        if count < max_count:
            needed = max_count - count
            class_indices = np.where(labels == class_label)[0]

            for _ in range(needed):
                selected_idx = np.random.choice(class_indices)

                original_key = keys[selected_idx]

                key_counts[original_key] += 1
                new_key = f"{original_key}_ovs_{key_counts[original_key]}"

                new_keys.append(new_key)
                new_features.append(features[selected_idx])
                new_labels.append(class_label)

    new_file_node_types = dict(zip(new_keys, new_features))

    return new_file_node_types, new_labels

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
            print('This file in csv list is not in project:' + handcraft_file_name)

    print("data size : " + str(count))
    return result

def ast_parse_CPDP(source_file_path):
    with open(source_file_path, 'rb') as file_obj:
        content = file_obj.read()
        result = []
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
                result.append(node_type)
                print(f"Node Type: {node_type}")
                print('-----------')
        print(result)
        return result

REGENERATE = False
dump_data_path ='../data_loader/autodl-tmp/'
if not os.path.exists(dump_data_path):
    os.makedirs(dump_data_path)

# 项目信息
root_path_source = '../data/projects/'
root_path_csv = '../data/csvs/'
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']

IMBALANCE_PROCESSOR = RandomOverSampler()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Analyze projects
path_projects = []
with open('../data/pairs-pre1.txt', 'r') as file_obj:
    for line in file_obj:
        stripped_line = line.strip()
        if stripped_line:
            path_projects.append(stripped_line)


for project in path_projects:

    path_source = root_path_source + project
    path_handcraft = root_path_csv + project + '.csv'
    project_name = path_source.rstrip('/').split('/')[-1]
    path_set = dump_data_path + project_name

    # If you don't need to regenerate, get it directly from dump_data
    if os.path.exists(path_set) and not REGENERATE:
        print("Loading saved data")
        obj = load_data(path_set)
        [train_graphs, test_graphs, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size] = obj
    else:
        # Get a list of instances of the training and test sets
        print("regenerate")
        file_instances = extract_handcraft_instances(path_handcraft)

        file_node_type = parse_source(path_source, file_instances, package_heads)

        all_node_types = set()
        for node_list in file_node_type.values():
            all_node_types.update(node_list)

        sorted_node_types = sorted(all_node_types)

        label = extract_label(path_handcraft, file_node_type).flatten()

        file_node_types, labels = random_oversample(file_node_type, label)

        file_features = {
            file: Counter(node_types)
            for file, node_types in file_node_types.items()
        }

        df = pd.DataFrame.from_dict(file_features, orient='index').fillna(0)

        df.to_csv(f'../data_dump/autodl-tmp/{project_name}_features_matrix.csv')

        min_total_freq = 3
        min_class_freq = 2

        df = df.loc[:, df.sum() >= min_total_freq]
        y = pd.Series(labels, index=df.index)

        valid_nodes = df.columns[
            (df[y == 1].sum() >= min_class_freq) |
            (df[y == 0].sum() >= min_class_freq)
            ]

        df_filtered = df[valid_nodes]

        df_binary = df_filtered.applymap(lambda x: 1 if x > 0 else 0)

        no_significant_nodes = []
        significant_nodes = []
        alpha = 0.05  # 原始显著性水平

        test_count = 0
        for node_type in df_filtered.columns:
            contingency_table = pd.crosstab(df_binary[node_type], y)
            if contingency_table.shape == (2, 2):
                test_count += 1
        adjusted_alpha = alpha / test_count

        for node_type in df_filtered.columns:

            contingency_table = pd.crosstab(df_binary[node_type], y)

            if contingency_table.shape != (2, 2):
                continue

            odds_ratio, p_value = fisher_exact(
                contingency_table,
                alternative='greater'

            )

            if p_value < adjusted_alpha or odds_ratio >1 :
                significant_nodes.append(node_type)
            else:
                no_significant_nodes.append(node_type)

        _, vocabulary = load_data(os.path.join('../data_loader/dic/', 'global_vocabulary.pkl'))
        indices = [vocabulary[node] for node in sorted_node_types]

        dump_data(os.path.join(dump_data_path, f'{project_name}_indices01.pkl'), indices)

        print(indices)