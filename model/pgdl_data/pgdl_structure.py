import os

pgdl_folderpath_prefixes = {
    1: "public_data",
    2: "public_data",
    4: "phase_one_data",
    5: "phase_one_data",
    6: "phase_two_data",
    7: "phase_two_data",
    8: "phase_two_data",
    9: "phase_two_data",
}

pgdl_folder_names = {
    1: "task1_v4",
    2: "task2_v1",
    4: "task4",
    5: "task5",
    6: "task6",
    7: "task7",
    8: "task8",
    9: "task9",
}


def get_task_data_location(pgdl_folderpath: str, task_number: int):
    pgdl_folderpath_prefix = pgdl_folderpath_prefixes[task_number]
    pgdl_folder_name = pgdl_folder_names[task_number]
    return os.path.join(pgdl_folderpath, pgdl_folderpath_prefix, 'input_data', pgdl_folder_name)


def get_task_metadata_filepath(pgdl_folderpath: str, task_number: int):
    pgdl_folderpath_prefix = pgdl_folderpath_prefixes[task_number]
    pgdl_folder_name = pgdl_folder_names[task_number]
    return os.path.join(pgdl_folderpath, pgdl_folderpath_prefix, 'reference_data', pgdl_folder_name,
                        'model_configs.json')
