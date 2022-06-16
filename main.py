from git import Repo
import projects
from correlation_exctractor import analyze_project, concat_project_stats_to_results
from meta_analysis import analyze_language_results, generate_stats_matrices
from settings import REPO_DIR, RESULTS_DIR, LANGUAGES, SOURCE_METER, STATS_DIR
import os

from utils import remove_empty_vectors, create_folder


def delete_repo(m_dir=REPO_DIR):
    import shutil
    import os
    import stat
    from os import path
    for root, dirs, files in os.walk(m_dir):
        for directory in dirs:
            os.chmod(path.join(root, directory), stat.S_IRWXU)
        for file in files:
            os.chmod(path.join(root, file), stat.S_IRWXU)
    try:
        shutil.rmtree(m_dir)
    except:
        print(f"Cannot delete folder {m_dir}")


def get_project_name(url):
    if url[-1] == '/':
        url = url[0:-1]
    parts = url.split('/')[-2:]
    return "__".join(parts)


def examine_projects(lang):
    delete_repo()
    for url in projects.SP_URLS[lang]:
        project_name = get_project_name(url)
        print(f"Start cloning project: {project_name}")
        Repo.clone_from(url, REPO_DIR, single_branch=True)
        print(f"End cloning project: {project_name}")

        additional_params = {
            lang == LANGUAGES[0]: [],
            lang == LANGUAGES[1]: [],
            lang == LANGUAGES[2]:
                ['-pythonVersion:3',
                 '-pythonBinary:C:/Users/supersloy/AppData/Local/Programs/Python/Python310/python']
        }[True]

        import subprocess
        try:
            subprocess.call([SOURCE_METER[lang],
                             f'-projectBaseDir:{REPO_DIR}',
                             f'-projectName:{project_name}',
                             f'-resultsDir:{RESULTS_DIR}/{lang}/{project_name}'] + additional_params,
                            timeout=1200)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT {project_name}")
        finally:
            delete_repo()


def analyze_projects(lang):
    res_dir = f'./{RESULTS_DIR}/{lang}'
    stats_dir = f'./{STATS_DIR}/{lang}'
    create_folder(stats_dir)  # create folder if there is none
    for subdir_name in next(os.walk(res_dir))[1]:
        print(f"Analyzing project: {subdir_name}")
        analyze_project(f"{res_dir}/{subdir_name}", subdir_name, lang, debug=False)
        concat_project_stats_to_results(f"{stats_dir}/{subdir_name}", lang)


if __name__ == '__main__':
    print("Start...")
    lang = LANGUAGES[1]
    # examine_projects(lang)
    analyze_projects(lang)
    analyze_language_results(lang)
    generate_stats_matrices(lang)
    remove_empty_vectors(lang)