import os
from pathlib import Path

def main(path_to_folder):
    # Получаем список всех файлов в данной папке
    files = list(Path(path_to_folder).glob('*'))

    for file in files:
        # Проверяем, начинается ли название файла с цифры
        if file.name[0].isdigit():
            try:
                number = int(file.name[:file.name.find('_')])
            except ValueError:
                file_name = file.stem
                number = int(file_name)

            # Создаем папку "Урок {число}" если её нет
            destination_folder = f'Урок {number}'
            if not os.path.exists(destination_folder):
                os.mkdir(destination_folder)

            # Перемещаем файл в соответствующую папку
            os.replace(str(file), os.path.join(destination_folder, file.name))

if __name__ == '__main__':
    # folder_path = input("Введите путь к папке: ")
    folder_path = Path("__file__").parent / "ipynb"
    main(folder_path)
