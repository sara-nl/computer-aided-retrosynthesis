import os


def reformat_data(source_file_name, target_file_name, out_file_name):

    with open(source_file_name, 'r') as source_file:
        source_lines = source_file.readlines()

    with open(target_file_name, 'r') as target_file:
        target_lines = target_file.readlines()

    assert len(source_lines) == len(target_lines)

    sequences = []

    for source, target in zip(source_lines, target_lines):
        source = source.strip()
        target = target.strip()

        source = ''.join(source.split())
        target = ''.join(target.split())

        sequence = source + ' >> ' + target
        sequences.append(sequence)

    with open(out_file_name, 'w') as f:
        for item in sequences:
            f.write("%s\n" % item)


if __name__ == '__main__':
    reformat_data('380k_products.txt', '380k_reactants.txt', '380k_dataset.smi')