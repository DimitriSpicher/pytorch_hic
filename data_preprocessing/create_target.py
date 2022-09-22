import os

# this scripts takes the images from separate folders, moves them into one folder and
# concatenates the subfolder name to the image


def rename_file(image_path):
    split = os.path.split(image_path)
    new_name = f'{split[0]}_{split[1]}'
    os.rename(image_path, new_name)
    return


def rename_files(sub_folder):
    files = os.listdir(sub_folder)

    for file in files:
        image_path = os.path.join(sub_folder, file)
        rename_file(image_path)
    return


if __name__ == '__main__':

    main_dir = r"D:\python\Pytorch_HIC\CIFAR-10_renamed\train"
    folders = os.listdir(main_dir)

    for folder in folders:
        sub_folder = os.path.join(main_dir, folder)
        rename_files(sub_folder)


