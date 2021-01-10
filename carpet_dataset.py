import csv

def get_indices(labels):
    indices = []
    with open('classic_handmade_digikala_labeled.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
           for l1, l2 in labels:
               if row[4] == l1 and row[5] == l2[1]:
                   indices.append(row[3])
                   break
        return indices



if __name__ == '__main__':
    indices = get_indices([[4, 'h']])
    print(indices)
