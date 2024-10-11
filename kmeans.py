import random
import matplotlib.pyplot as plt
from collections import defaultdict
from random import uniform
from math import sqrt
import numpy as np

def shuffle_data(data, labels):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return data, labels

def point_avg(points):
    dimensions = len(points[0])
    new_center = []
    for dimension in range(dimensions):
        dim_sum = 0
        for p in points:
            dim_sum += p[dimension]
        new_center.append(dim_sum / float(len(points)))
    return new_center

def update_centers(data_set, assignments):
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
    for points in new_means.values():
        centers.append(point_avg(points))
    return centers

def assign_points(data_points, centers):
    assignments = []
    for point in data_points:
        shortest = float('inf')
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments

def distance(a, b):
    dimensions = len(a)
    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)

def generate_k(data_set, k):
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)
    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val
    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            if i == 0:  # An
                rand_point.append(random.randint(int(min_val), int(max_val)))
            elif i == 1:  # Luna
                rand_point.append(random.randint(1, 12))
            elif i == 2:  # Zi
                rand_point.append(random.randint(1, 31))
            else:
                rand_point.append(uniform(min_val, max_val))
        centers.append(rand_point)
    return centers

def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return list(zip(assignments, dataset)), new_centers

def plot_clusters(clusters, centers):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for idx, cluster in enumerate(clusters):
        points = [point for assignment, point in clusters if assignment == idx]
        x_coords = [point[4] for point in points]
        y_coords = [point[5] for point in points]
        plt.scatter(x_coords, y_coords, c=colors[idx % len(colors)], label=f'Cluster {idx + 1}')
    center_x_coords = [center[4] for center in centers]
    center_y_coords = [center[5] for center in centers]
    plt.scatter(center_x_coords, center_y_coords, c='black', marker='x', s=100, label='Centers')
    plt.xlabel('Monoxid de carbon')
    plt.ylabel('Dioxid de sulf')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.show()

def calculate_feature_importance(data, labels):
    features = np.array(data)
    variances = np.var(features, axis=0)
    importance = variances / np.sum(variances)
    return importance

def print_feature_importance(importance):
    attributes = [
        "an", "luna", "zi", "ora",
        "monoxid de carbon", "dioxid de sulf", "particule de suspensie", "dioxid de azot"
    ]
    print("Importanța atributelor pentru setul de date 1:")
    print("\t" + "\t".join(attributes))
    print("Importanță:\t" + "\t".join(f"{imp:.4f}" for imp in importance))

def gini_impurity(labels):
    total = len(labels)
    if total == 0:
        return 0
    counts = np.bincount(labels)
    probabilities = counts / total
    return 1 - np.sum(probabilities ** 2)

def split_data(data, labels, feature_index, threshold):
    left_data, right_data = [], []
    left_labels, right_labels = [], []
    for i in range(len(data)):
        if data[i][feature_index] <= threshold:
            left_data.append(data[i])
            left_labels.append(labels[i])
        else:
            right_data.append(data[i])
            right_labels.append(labels[i])
    return (left_data, left_labels), (right_data, right_labels)

def find_best_split(data, labels, criterion='gini'):
    best_metric = float('inf') if criterion == 'gini' else -float('inf')
    best_split = None
    for feature_index in range(len(data[0])):
        thresholds = sorted(set(row[feature_index] for row in data))
        for threshold in thresholds:
            (left_data, left_labels), (right_data, right_labels) = split_data(data, labels, feature_index, threshold)
            if criterion == 'gini':
                gini_left = gini_impurity(left_labels)
                gini_right = gini_impurity(right_labels)
                gini = (len(left_labels) * gini_left + len(right_labels) * gini_right) / len(labels)
                if gini < best_metric:
                    best_metric = gini
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left': (left_data, left_labels),
                        'right': (right_data, right_labels),
                        'gini': gini
                    }
    return best_split

def build_decision_tree(data, labels, depth=0, max_depth=3, criterion='gini'):
    if depth >= max_depth or len(set(labels)) == 1:
        return {'type': 'leaf', 'class': max(set(labels), key=labels.count)}
    split = find_best_split(data, labels, criterion)
    if not split:
        return {'type': 'leaf', 'class': max(set(labels), key=labels.count)}
    left_tree = build_decision_tree(*split['left'], depth + 1, max_depth, criterion)
    right_tree = build_decision_tree(*split['right'], depth + 1, max_depth, criterion)
    return {
        'type': 'node',
        'feature_index': split['feature_index'],
        'threshold': split['threshold'],
        'left': left_tree,
        'right': right_tree,
        'metric': split.get('gini'),
        'samples': len(labels),
        'value': [labels.count(i) for i in range(1, 5)]
    }

def tree_to_graphviz(tree, node_id=0, depth=0):
    if tree['type'] == 'leaf':
        return f'    {node_id} [label="class = {tree["class"]}", fillcolor="#e58139ff"] ;\n'
    node_label = f'X<SUB>{tree["feature_index"]}</SUB> &le; {tree["threshold"]}<br/>{tree.get("metric", "")} = {tree["metric"]}<br/>samples = {tree["samples"]}<br/>value = {tree["value"]}'
    node_code = f'    {node_id} [label=<{node_label}>, fillcolor="#e58139a4"] ;\n'
    left_id = node_id * 2 + 1
    right_id = node_id * 2 + 2
    left_code = tree_to_graphviz(tree['left'], left_id, depth + 1)
    right_code = tree_to_graphviz(tree['right'], right_id, depth + 1)
    edge_code = f'    {node_id} -> {left_id} [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n'
    edge_code += f'    {node_id} -> {right_id} [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n'
    return node_code + left_code + right_code + edge_code

def generate_decision_tree_code(data, labels, max_depth=3, criterion='gini'):
    tree = build_decision_tree(data, labels, max_depth=max_depth, criterion=criterion)
    graphviz_code = f"""
    digraph Tree {{
    node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;
    edge [fontname=helvetica] ;
    {tree_to_graphviz(tree)}
    }}
    """
    return graphviz_code

def save_graphviz_code(file_path, data, labels, criterion='gini'):
    graphviz_code = generate_decision_tree_code(data, labels, criterion=criterion)
    with open(file_path, 'w') as file:
        file.write(graphviz_code)
