import csv

csv_path = 'classic_handmade_digikala_labeled_simple.csv'
labels_to_int = {'l':0,
                'm':1,
                'k':2,
                't':3,
                'g':4,
                'h':5}


def get_indices(labels):
    """
        get index of file names with labels like 4 l, etc.
        Usage: indices = get_indices([['4', 'h'], ['4', 'l'], ['4', '*']])
    """
    indices = []
    with open(csv_path) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            for label in labels:
                l1, l2 = label
                if (row[1] == l1 and row[2] == l2) or (row[1] == l1 and l2 == '*') or (l1 == '*' and row[2] == l2) or (l1 == '*' and l2 == '*'):
                    indices.append(row[0])
        return indices


def get_labels(indices):
    """
        get label of given indices as dictionary
    """
    indices = [adrs.split('\\')[-1] for adrs in indices]
    inx_labels = {}
    labels = []
    # for index in indices:
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if row[0] in indices:
                inx_labels[row[0]] = labels_to_int[row[2]]
                labels.append(labels_to_int[row[2]])
        # return inx_labels
        return labels

if __name__ == '__main__':
    # # indices = get_indices([['4', 'h'], ['4', 'l']])
    # indices = get_indices([['*', 'h']])
    # print(len(indices))

    # indices = [str(i)+'.jpg' for i in range(20)]
    # print(get_labels(indices))

    pass
