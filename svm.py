import random
import matplotlib.pyplot as plt
import numpy as np

def shuffle_data(data, labels):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return data, labels

def filter_data(data, labels):
    filtered_data = []
    filtered_labels = []
    for i in range(len(data)):
        an, luna, zi = data[i][0], data[i][1], data[i][2]
        if an != 0 and luna != 0 and zi != 0:
            filtered_data.append(data[i])
            filtered_labels.append(labels[i])
    return filtered_data, filtered_labels

class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.array(y)
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

def plot_decision_boundary(data, labels, model):
    data = np.array(data)
    labels = np.array(labels)
    h = .02

    x_min, x_max = data[:, 6].min() - 1, data[:, 6].max() + 1
    y_min, y_max = data[:, 5].min() - 1, data[:, 5].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    color_mapping = {1: 'red', 2: 'yellow', 3: 'grey'}
    scatter = plt.scatter(data[:, 6], data[:, 5], c=[color_mapping[label] for label in labels], edgecolors='k')
    plt.xlabel('Particule de suspensie')
    plt.ylabel('Dioxid de sulf')
    plt.title('SVM: kernel liniar')
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

def find_best_split(data, labels):
    best_gini = float('inf')
    best_split = None
    for feature_index in range(len(data[0])):
        thresholds = sorted(set(row[feature_index] for row in data))
        for threshold in thresholds:
            (left_data, left_labels), (right_data, right_labels) = split_data(data, labels, feature_index, threshold)
            gini_left = gini_impurity(left_labels)
            gini_right = gini_impurity(right_labels)
            gini = (len(left_labels) * gini_left + len(right_labels) * gini_right) / len(labels)
            if gini < best_gini:
                best_gini = gini
                best_split = {
                    'feature_index': feature_index,
                    'threshold': threshold,
                    'left': (left_data, left_labels),
                    'right': (right_data, right_labels),
                    'gini': gini
                }
    return best_split

def build_decision_tree(data, labels, depth=0, max_depth=3):
    if depth >= max_depth or len(set(labels)) == 1:
        return {'type': 'leaf', 'class': max(set(labels), key=labels.count)}
    split = find_best_split(data, labels)
    if not split:
        return {'type': 'leaf', 'class': max(set(labels), key=labels.count)}
    left_tree = build_decision_tree(*split['left'], depth + 1, max_depth)
    right_tree = build_decision_tree(*split['right'], depth + 1, max_depth)
    return {
        'type': 'node',
        'feature_index': split['feature_index'],
        'threshold': split['threshold'],
        'left': left_tree,
        'right': right_tree,
        'gini': split['gini'],
        'samples': len(labels),
        'value': [labels.count(i) for i in range(1, 5)]
    }

def tree_to_graphviz(tree, node_id=0, depth=0):
    if tree['type'] == 'leaf':
        return f'    {node_id} [label="class = {tree["class"]}", fillcolor="#e58139ff"] ;\n'
    node_label = f'X<SUB>{tree["feature_index"]}</SUB> &le; {tree["threshold"]}<br/>gini = {tree["gini"]}<br/>samples = {tree["samples"]}<br/>value = {tree["value"]}'
    node_code = f'    {node_id} [label=<{node_label}>, fillcolor="#e58139a4"] ;\n'
    left_id = node_id * 2 + 1
    right_id = node_id * 2 + 2
    left_code = tree_to_graphviz(tree['left'], left_id, depth + 1)
    right_code = tree_to_graphviz(tree['right'], right_id, depth + 1)
    edge_code = f'    {node_id} -> {left_id} [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n'
    edge_code += f'    {node_id} -> {right_id} [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n'
    return node_code + left_code + right_code + edge_code

def generate_decision_tree_code(data, labels, max_depth=3):
    tree = build_decision_tree(data, labels, max_depth=max_depth)
    graphviz_code = f"""
    digraph Tree {{
    node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;
    edge [fontname=helvetica] ;
    {tree_to_graphviz(tree)}
    }}
    """
    return graphviz_code

def save_graphviz_code(file_path, data, labels):
    graphviz_code = generate_decision_tree_code(data, labels)
    with open(file_path, 'w') as file:
        file.write(graphviz_code)
