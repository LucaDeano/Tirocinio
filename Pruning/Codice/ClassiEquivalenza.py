import ast
from collections import Counter
import matplotlib.pyplot as plt

def count_equivalence_classes(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        data = ast.literal_eval(content)
    
    total_classes = 0
    layer_counts = {}
    class_sizes = []

    for layer, layer_data in data.items():
        if isinstance(layer_data, dict):
            layer_count = len(layer_data)
            total_classes += layer_count
            layer_counts[layer] = layer_count
            
            for eq_class in layer_data.values():
                if isinstance(eq_class, list):
                    class_sizes.append(len(eq_class))

    return total_classes, layer_counts, class_sizes

def main():
    file_path = 'lumped_neurons.txt'
    total_classes, layer_counts, class_sizes = count_equivalence_classes(file_path)

    print(f"Total number of equivalence classes: {total_classes}")
    print("\nNumber of equivalence classes per layer:")
    for layer, count in layer_counts.items():
        print(f"Layer {layer}: {count} classes")

    if class_sizes:
        print(f"\nLargest equivalence class size: {max(class_sizes)}")
        print(f"Smallest equivalence class size: {min(class_sizes)}")
        print(f"Average equivalence class size: {sum(class_sizes) / len(class_sizes):.2f}")

        # Count the frequency of each class size
        size_counter = Counter(class_sizes)
        most_common_size, frequency = size_counter.most_common(1)[0]
        print(f"\nMost common equivalence class size: {most_common_size} (occurs {frequency} times)")

    else:
        print("\nNo valid class sizes found.")

if __name__ == "__main__":
    main()