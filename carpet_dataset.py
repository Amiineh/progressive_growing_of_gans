import csv

def get_indices(labels):
    indices = []
    with open('classic_handmade_digikala_labeled_simple.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            for label in labels:
                l1, l2 = label
                if row[1] == l1 and row[2] == l2:
                    indices.append(row[0])
        return indices


if __name__ == '__main__':
    indices = get_indices([['4', 'h'], ['4', 'l']])
    print(indices)
