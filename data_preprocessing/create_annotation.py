# in test and train folder create csv file with path class and annotation
import os
import pandas as pd
import numpy as np


def create_annotation(lineage, categories):
    target_vector = []
    for cat in categories:
        target = cat in lineage
        target_vector.append(target)

    target_vector = list(map(int, target_vector))

    return np.array(target_vector)


def get_lineage(image_name, lineages):

    category = image_name.split('_')[0]
    for lin in lineages:
        if category in lin:
            break

    return '-'.join(lin)


def main(image_path, lineages, column_names, categories):
    image_names = os.listdir(image_path)

    lins = []
    targets = []
    for name in image_names:
        lin = get_lineage(name, lineages)
        target_vector = create_annotation(lin, categories)
        lins.append(lin)
        targets.append(target_vector)

    annotations = pd.concat([pd.DataFrame(image_names),
                             pd.DataFrame(lins), pd.DataFrame(targets)],
                            axis=1, ignore_index=True)

    annotations.columns = column_names
    return annotations


if __name__ == "__main__":
    main_category = ['object', 'animal']
    sub_category_1 = ['driving', 'not', 'mammal', 'other']
    sub_category_2 = ['truck', 'automobile', 'ship', 'airplane', 'deer', 'dog', 'cat', 'horse', 'frog', 'bird']

    categories = main_category + sub_category_1 + sub_category_2
    column_names = ['id', 'lineage'] + categories
    lineages = [['object', 'driving', 'truck'], ['object', 'driving', 'automobile'],
                ['object', 'not', 'ship'], ['object', 'not', 'airplane'], ['animal', 'mammal', 'deer'],
                ['animal', 'mammal', 'dog'], ['animal', 'mammal', 'cat'], ['animal', 'mammal', 'horse'],
                ['animal', 'other', 'frog'], ['animal', 'other', 'bird']]

    image_path = r"D:\python\Pytorch_HIC\CIFAR-10_renamed\image_folder"

    annotations = main(image_path, lineages, column_names, categories)

    annotations.to_csv(r'D:\python\Pytorch_HIC\CIFAR-10_renamed\annotations.csv', index=False)