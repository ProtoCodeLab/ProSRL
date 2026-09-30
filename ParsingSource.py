#import tool.javalang as jl
import javalang as jl
import os
import javalang.tree as jlt#tool.javalang.tree as jlt
import numpy as np
import pandas as pd
import networkx as nx
#from tool import javalang
import javalang
import dgl
import torch

features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam',
            'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']


def extract_data_and_generate_graphs(project_root_path, balanced_file_instances, package_heads, graph_vocabulary = None):
    results = []
    count = 0
    missing_count = 0

    for qualified_name in balanced_file_instances:

        matched = False
        for dir_path, _, file_names in os.walk(project_root_path):
            index = -1
            for _head in package_heads:
                index = dir_path.find(_head)
                if index >= 0:
                    break
            if index < 0:
                continue

            package_name = dir_path[index:].replace(os.sep, '.')

            for file in file_names:
                if file.endswith('.java') and f"{package_name}.{file}" == qualified_name:
                    full_path = os.path.join(dir_path, file)
                    graph = parse_java_to_graph(full_path)
                    if graph:
                        dgl_graph = convert_to_dgl(graph, vocabulary = graph_vocabulary)
                        if dgl_graph:
                            results.append((qualified_name, dgl_graph))
                            count += 1
                        else:
                            missing_count += 1
                    else:
                        missing_count += 1
                    matched = True
                    break

            if matched:
                break

        if not matched:
            print(f"This file is not in directory structure: {qualified_name}")
            missing_count += 1

    print(f"Total graphs generated: {count}")
    print(f"Missing graphs for files: {missing_count}")
    return results

def parse_java_to_graph(source_file_path):
    with open(source_file_path, 'rb') as file_obj:
        content = file_obj.read()
    try:
        tree = javalang.parse.parse(content)
    except javalang.parser.JavaSyntaxError as e:
        print(f'JavaSyntaxError in file {source_file_path}: {e.description} at {e.at}')
        return None

    graph = nx.DiGraph()
    node_counter = 0

    def add_nodes_edges(node, parent_id=None):
        nonlocal node_counter

        current_id = node_counter
        node_counter += 1

        graph.add_node(current_id, type=type(node).__name__)

        if parent_id is not None:
            graph.add_edge(parent_id, current_id)

        if hasattr(node, 'children'):
            for child in node.children:
                if isinstance(child, (list, tuple)):
                    for c in child:
                        if isinstance(c, javalang.tree.Node):
                            add_nodes_edges(c, current_id)
                elif isinstance(child, javalang.tree.Node):
                    add_nodes_edges(child, current_id)

    add_nodes_edges(tree)
    return graph

def convert_to_dgl(graph, vocabulary=None):

    if not graph:
        return None

    node_types = [graph.nodes[node]['type'] for node in graph.nodes]

    if vocabulary is not None:
        features = [vocabulary.get(node_type, 0) for node_type in node_types]
    else:
        features = [0] * len(node_types)

    features = torch.tensor(features, dtype=torch.int64)

    assert len(features) == graph.number_of_nodes(), "Mismatch between number of features and number of nodes"

    dgl_graph = dgl.from_networkx(graph)
    dgl_graph.ndata['feat'] = features

    dgl_graph = dgl.add_self_loop(dgl_graph)

    if torch.cuda.is_available():
        dgl_graph = dgl_graph.to('cuda')
        dgl_graph.ndata['feat'] = dgl_graph.ndata['feat'].to('cuda')

    print(f"Graph info: {dgl_graph.num_nodes()} nodes, {dgl_graph.num_edges()} edges")

    return dgl_graph

def append_suffix(df):
    for i in range(len(df['file_name'])):
        df.loc[i, 'file_name'] = df.loc[i, 'file_name'] + ".java"
    return df

def extract_handcraft_instances(path):
    handcraft_instances = pd.read_csv(path)
    handcraft_instances = append_suffix(handcraft_instances)
    handcraft_instances = np.array(handcraft_instances['file_name'])
    handcraft_instances = handcraft_instances.tolist()

    return handcraft_instances

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
        # print(result)
        return result

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
            print('This file is not in csv list:' + handcraft_file_name)

    print("data size : " + str(count))
    return result


def padding_vector(vector, size):
    if len(vector) >= size:
        return vector[:size]

    pad_length = size - len(vector)
    padding = np.zeros(pad_length).tolist()
    vector += padding
    return vector


def padding_all(dict_token, size):
    result = {}
    for key, vector in dict_token.items():
        pv = padding_vector(vector, size)
        result[key] = pv
    return result


def max_length(d):
    max_len = 0
    for value in d.values():
        if max_len < len(value):
            max_len = len(value)
    return max_len

def transform_token_to_number(list_dict_token):
    frequence = {}
    for _dict_token in list_dict_token:
        for _token_vector in _dict_token.values():
            for _token in _token_vector:
                if frequence.__contains__(_token):
                    frequence[_token] = frequence[_token] + 1
                else:
                    frequence[_token] = 1

    vocabulary = {}
    result = []
    count = 0
    max_len = 0
    for dict_token in list_dict_token:
        _dict_encode = {}
        for file_name, token_vector in dict_token.items():
            vector = []
            for v in token_vector:
                if frequence[v] < 3:
                    continue

                if vocabulary.__contains__(v):
                    vector.append(vocabulary.get(v))
                else:
                    count = count + 1
                    vector.append(count)
                    vocabulary[v] = count
            if len(vector) > max_len:
                max_len = len(vector)
            _dict_encode[file_name] = vector
        result.append(_dict_encode)

    for i in range(len(result)):
        result[i] = padding_all(result[i], max_len)
    return result, max_len, len(vocabulary),vocabulary


def extract_data(path_handcraft_file, dict_encoding_vector):

    def extract_label(df, file_name):
        row = df[df.file_name == file_name]['bug']
        row = np.array(row).tolist()
        if row[0] > 1:
            row[0] = 1
        return row

    def extract_feature(df, file_name):
        row = df[df.file_name == file_name][features]
        row = np.array(row).tolist()
        row = np.squeeze(row)
        row = list(row)
        return row

    ast_x_data = []
    hand_x_data = []
    label_data = []
    raw_handcraft = pd.read_csv(path_handcraft_file)
    raw_handcraft = append_suffix(raw_handcraft)
    for key, value in dict_encoding_vector.items():
        ast_x_data.append(value)

        hand_x_data.append(extract_feature(raw_handcraft, key))
        label_data.append(extract_label(raw_handcraft, key))
    ast_x_data = np.array(ast_x_data)
    hand_x_data = np.array(hand_x_data)
    label_data = np.array(label_data)

    return ast_x_data, hand_x_data, label_data

def extract_label(path_handcraft_file, dict_encoding_vector):

    def extract_label(df, file_name):
        row = df[df.file_name == file_name]['bug']
        row = np.array(row).tolist()
        if row[0] > 1:
            row[0] = 1
        return row

    def extract_feature(df, file_name):
        row = df[df.file_name == file_name][features]
        row = np.array(row).tolist()
        row = np.squeeze(row)
        row = list(row)
        return row

    label_data = []
    raw_handcraft = pd.read_csv(path_handcraft_file)
    raw_handcraft = append_suffix(raw_handcraft)
    for key, value in dict_encoding_vector.items():

        label_data.append(extract_label(raw_handcraft, key))
    label_data = np.array(label_data)

    return label_data