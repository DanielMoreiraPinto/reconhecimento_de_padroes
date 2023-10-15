import os

path, _ = os.path.split(os.path.abspath(__file__))
folder_relative_path = path.split(os.sep)[-1]
absolute_root_path = path
project_folder_path = "\\".join(path.split(os.sep)[:-1])
# descomenta aqui se quiser ver
# print(folder_relative_path)
# print(absolute_root_path)
# print('teste',  project_folder_path)
