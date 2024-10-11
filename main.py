import random
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import kmeans
import svm
import matplotlib.pyplot as plt

def generate_random_data(num_samples):
    data = []
    labels = []
    for _ in range(num_samples):
        an = random.randint(2010, 2024)
        luna = random.randint(1, 12)
        zi = random.randint(1, 31)
        ora = random.randint(0, 23)
        monoxid_carbon = random.uniform(0, 10)
        dioxid_sulf = random.uniform(0, 100)
        particule_suspensie = random.uniform(0, 50)
        dioxid_azot = random.uniform(0, 30)
        label = random.randint(1, 3)  # Scor 1, 2 sau 3
        data.append([an, luna, zi, ora, monoxid_carbon, dioxid_sulf, particule_suspensie, dioxid_azot])
        labels.append(label)
    return data, labels

def display_data_table(data, labels, tree):
    for row in tree.get_children():
        tree.delete(row)
    for row in data:
        tree.insert("", "end", values=row)


def plot_feature_importance(importance):
    attributes = ["monoxid de carbon", "dioxid de sulf", "particule de suspensie", "dioxid de azot"]
    relevant_importance = importance[4:]  # select the relevant features

    colors = ['red', 'green', 'blue', 'orange']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(attributes, relevant_importance, color=colors)
    plt.ylabel('Importanță GINI')
    plt.title('Importanța atributelor pentru setul de date')
    plt.legend(bars, attributes, loc='upper right')

    # Eliminăm numerele de sub axa x
    plt.gca().get_xaxis().set_ticks([])

    plt.show()


def plot_decision_boundary(data, labels, model):
    data = np.array(data)
    labels = np.array(labels)
    h = .02

    x_min, x_max = data[:, 3].min() - 1, data[:, 3].max() + 1  # Ora
    y_min, y_max = data[:, 7].min() - 1, data[:, 7].max() + 1  # NO2 (Dioxid de azot)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(16, 10))  # Am mărit dimensiunea figurii pentru mai mult spațiu
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Adăugăm linii de contur pentru limitele deciziei
    plt.contour(xx, yy, Z, colors='k', linewidths=2, linestyles='dashed')

    color_mapping = {1: 'red', 2: 'orange', 3: 'grey'}
    scatter = plt.scatter(data[:, 3], data[:, 7], c=[color_mapping[label] for label in labels], edgecolors='k', s=100, alpha=0.9)

    # Adăugăm legenda pentru culorile punctelor, cu explicații pentru clase
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Clasa 1: Nivel scăzut de poluare'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Clasa 2: Nivel moderat de poluare'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Clasa 3: Nivel ridicat de poluare')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.4, 0.5), ncol=1)

    plt.xlabel('Ora')
    plt.ylabel('NO2')
    plt.title('SVC with linear kernel')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(scatter, ax=plt.gca(), label='Classes')

    # Adăugăm anotări pentru puncte cheie (opțional)
    for i in range(len(data)):
        plt.text(data[i, 3], data[i, 7], f'{labels[i]}', fontsize=9, ha='right')

    plt.show()






def run_algorithm(algorithm, run_count, num_samples, tree, importance_tree):
    for run in range(run_count):
        data, labels = generate_random_data(num_samples)
        display_data_table(data, labels, tree)

        if algorithm == 'K-Means':
            data, labels = kmeans.shuffle_data(data, labels)
            k = 4
            clusters, centers = kmeans.k_means(data, k)
            kmeans.plot_clusters(clusters, centers)
            importance_gini = kmeans.calculate_feature_importance(data, labels)
            kmeans.print_feature_importance(importance_gini)
            kmeans.save_graphviz_code(f'kmeans_decision_tree_run_{run + 1}.dot', data, labels, criterion='gini')

        elif algorithm == 'SVM':
            data, labels = svm.shuffle_data(data, labels)
            data, labels = svm.filter_data(data, labels)
            svm_model = svm.SimpleSVM()
            svm_model.fit(np.array(data)[:, [6, 5]], labels)  # Antrenare pe Particule de suspensie și Dioxid de sulf
            plot_decision_boundary(data, labels, svm_model)
            importance_gini = svm.calculate_feature_importance(data, labels)
            svm.print_feature_importance(importance_gini)
            svm.save_graphviz_code(f'svm_decision_tree_run_{run + 1}.dot', data, labels)

        display_feature_importance(importance_tree, importance_gini)
        plot_feature_importance(importance_gini)

def display_feature_importance(tree, gini_importance):
    for row in tree.get_children():
        tree.delete(row)

    # Atributele relevante pentru poluare
    attributes = ["monoxid de carbon", "dioxid de sulf", "particule de suspensie", "dioxid de azot"]

    # Display the relevant importance values for the attributes starting from index 4
    for attr, gini_imp in zip(attributes, gini_importance[4:]):
        tree.insert("", "end", values=(attr, f"{gini_imp:.4f}"))

def start():
    algorithm = algorithm_var.get()
    try:
        run_count = int(run_count_entry.get())
        num_samples = int(num_samples_entry.get())
    except ValueError:
        messagebox.showerror("Eroare", "Numărul de rulari și numărul de eșantioane trebuie să fie numere întregi.")
        return

    if algorithm not in ['K-Means', 'SVM']:
        messagebox.showerror("Eroare", "Selectează un algoritm valid.")
        return

    run_algorithm(algorithm, run_count, num_samples, data_tree, importance_tree)

root = ThemedTk(theme="radiance")
root.title("Algoritmi de Machine Learning")
root.configure(background='#ADD8E6')

style = ttk.Style()
style.configure('TFrame', background='#ADD8E6')
style.configure('TLabel', background='#ADD8E6', font=('Arial', 12))
style.configure('TRadiobutton', background='#ADD8E6', font=('Arial', 12))
style.configure('TButton', font=('Arial', 12, 'bold'))
style.configure('TEntry', font=('Arial', 12))
style.configure('Treeview', rowheight=25)
style.configure('Treeview.Heading', font=('Arial', 12, 'bold'))

# Definire stil personalizat pentru butonul "Start"
style.configure('Start.TButton', background='#4CAF50', foreground='black')

algorithm_var = tk.StringVar(value='K-Means')

# Creare frame pentru selecția algoritmului
frame_selection = ttk.Frame(root, padding="10")
frame_selection.grid(row=0, column=0, sticky=(tk.W, tk.E))

algorithm_label = ttk.Label(frame_selection, text="Selectează algoritmul:")
algorithm_label.grid(row=0, column=0, sticky=tk.W)

kmeans_radio = ttk.Radiobutton(frame_selection, text='K-Means', variable=algorithm_var, value='K-Means')
kmeans_radio.grid(row=1, column=0, sticky=tk.W)

svm_radio = ttk.Radiobutton(frame_selection, text='SVM', variable=algorithm_var, value='SVM')
svm_radio.grid(row=1, column=1, sticky=tk.W)

# Creare frame pentru input-ul utilizatorului
frame_input = ttk.Frame(root, padding="10")
frame_input.grid(row=1, column=0, sticky=(tk.W, tk.E))

run_count_label = ttk.Label(frame_input, text="Numărul de rulari:")
run_count_label.grid(row=0, column=0, sticky=tk.W)
run_count_entry = ttk.Entry(frame_input, width=7)
run_count_entry.grid(row=0, column=1, sticky=tk.W)

num_samples_label = ttk.Label(frame_input, text="Numărul de eșantioane:")
num_samples_label.grid(row=1, column=0, sticky=tk.W)
num_samples_entry = ttk.Entry(frame_input, width=7)
num_samples_entry.grid(row=1, column=1, sticky=tk.W)

# Creare frame pentru butonul de start
frame_start = ttk.Frame(root, padding="10")
frame_start.grid(row=2, column=0, sticky=(tk.W, tk.E))

start_button = ttk.Button(frame_start, text="Start", command=start, style='Start.TButton')
start_button.grid(row=0, column=0, sticky=tk.W)

# Creare frame pentru afișarea importanței atributelor
importance_frame = ttk.Frame(root, padding="10")
importance_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))

importance_tree = ttk.Treeview(importance_frame, columns=("Atribut", "Importanță GINI"), show='headings', height=10)
importance_tree.heading("Atribut", text="Atribut", anchor=tk.CENTER)
importance_tree.heading("Importanță GINI", text="Importanță GINI", anchor=tk.CENTER)
importance_tree.column("Atribut", anchor=tk.CENTER)
importance_tree.column("Importanță GINI", anchor=tk.CENTER)
importance_tree.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Adăugare scrollbar la widget-ul Treeview
importance_scrollbar = ttk.Scrollbar(importance_frame, orient=tk.VERTICAL, command=importance_tree.yview)
importance_tree.configure(yscroll=importance_scrollbar.set)
importance_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

# Creare frame pentru afișarea datelor
frame_output = ttk.Frame(root, padding="10")
frame_output.grid(row=4, column=0, sticky=(tk.W, tk.E))

# Creare Treeview pentru afișarea datelor
columns = ["An", "Luna", "Zi", "Ora", "Monoxid de Carbon", "Dioxid de Sulf", "Particule de Suspensie", "Dioxid de Azot"]
data_tree = ttk.Treeview(frame_output, columns=columns, show='headings', height=20)
for col in columns:
    data_tree.heading(col, text=col, anchor=tk.CENTER)
    data_tree.column(col, anchor=tk.CENTER)
data_tree.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Ajustăm lățimea coloanelor pentru a asigura vizibilitatea completă a numelor
data_tree.column("An", width=60)
data_tree.column("Luna", width=60)
data_tree.column("Zi", width=60)
data_tree.column("Ora", width=60)
data_tree.column("Monoxid de Carbon", width=120)
data_tree.column("Dioxid de Sulf", width=120)
data_tree.column("Particule de Suspensie", width=140)
data_tree.column("Dioxid de Azot", width=120)

# Adăugare scrollbar
scrollbar = ttk.Scrollbar(frame_output, orient=tk.VERTICAL, command=data_tree.yview)
data_tree.configure(yscroll=scrollbar.set)
scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

# Configurarea greutății coloanelor și rândurilor pentru a face interfața responsive
root.grid_columnconfigure(0, weight=1)
frame_selection.grid_columnconfigure(0, weight=1)
frame_input.grid_columnconfigure(0, weight=1)
frame_start.grid_columnconfigure(0, weight=1)
importance_frame.grid_columnconfigure(0, weight=1)
importance_frame.grid_rowconfigure(0, weight=1)
frame_output.grid_columnconfigure(0, weight=1)
frame_output.grid_rowconfigure(0, weight=1)

# Rulare interfață grafică
root.mainloop()
