import matplotlib.pyplot as plt
import sys


def conv_list_graph(input_file, graph_file):
    total, ok = 0, 0
    conv = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                _, res = int(parts[0]), float(parts[1])
                total += 1
                if res == 1:
                    ok += 1
                acc = ok / total * 100
                print(f'{total}\t{acc:.2f}')
                conv.append((total, acc))

    x = [x[0] for x in conv]
    y = [x[1] for x in conv]
    plt.plot(x, y)
    plt.xlabel('Nb of examples')
    plt.ylabel('Accuracy')
    plt.title('Convergence')
    plt.savefig(graph_file)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convergence_graph.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    graph_file = input_file + "_conv.png"
    conv_list_graph(input_file, graph_file)
