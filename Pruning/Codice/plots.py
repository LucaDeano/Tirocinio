import ast
from collections import Counter
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def count_equivalence_classes(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        data = ast.literal_eval(content)
    
    layer_counts = {}
    for layer, layer_data in data.items():
        if isinstance(layer_data, dict):
            layer_counts[layer] = len(layer_data)

    return layer_counts

def process_all_files(base_path):
    results = []
    architectures = ['300-100', '500-300', '500-300-100']
    batch_sizes = [80, 120, 160, 200, 240, 280, 320, 360, 400, 440]

    for arch in architectures:
        for batch in batch_sizes:
            file_path = os.path.join(base_path, arch, f'batch{batch}', f'lumped_neurons.txt')
            
            if os.path.exists(file_path):
                layer_counts = count_equivalence_classes(file_path)
                
                for layer, count in layer_counts.items():
                    results.append({
                        'Architecture': arch,
                        'Batch Size': batch,
                        'Layer': f'Hidden {layer}',
                        'Equivalence Classes': count
                    })
            else:
                print(f"File not found: {file_path}")

    return pd.DataFrame(results)

def plot_layer_means(df):
    plt.figure(figsize=(15, 8))
    
    sns.set_palette("icefire", n_colors=5)
    
    mean_equiv_classes = df[df['Layer'] != 'Hidden 0'].groupby(['Architecture', 'Layer'])['Equivalence Classes'].mean().reset_index()
    
    ax = sns.barplot(x='Architecture', y='Equivalence Classes', hue='Layer', data=mean_equiv_classes)
    
    plt.xlabel('Architecture', fontsize=14)
    plt.ylabel('Mean Number of Equivalence Classes', fontsize=14)
    
    # Move the legend inside the plot
    plt.legend(title='Layer', loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=14, title_fontsize=14)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=2, fontsize=16)
    
    plt.ylim(0, plt.ylim()[1] * 1.1)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig('equivalente_classes.png', dpi=300)
    plt.close()

def main():
    base_path = 'Reti'
    results_df = process_all_files(base_path)
    
    print(results_df)
    results_df.to_csv('LumpingPerLayer.csv', index=False)
    
    plot_layer_means(results_df)

if __name__ == "__main__":
    main()