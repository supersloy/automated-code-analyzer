REPO_DIR = "temp_repo_directory"
RESULTS_DIR = "results"
STATS_DIR = "stats"
STATS_RESULTS_DIR = "results"
LANGUAGES = ["JavaScript", "Java", "Python", "CSharp", "CPP"]

SOURCE_METER = {
    LANGUAGES[0]: "./JavaScript/SourceMeterJavaScript.exe",
    LANGUAGES[1]: "./Java/SourceMeterJava.exe",
    LANGUAGES[2]: "./Python/SourceMeterPython.exe",
    LANGUAGES[3]: "./CSharp/SourceMeterCSharp.exe",
    LANGUAGES[4]: "./CPP/SourceMeterCSharp.exe"
}

# def getSourceMeterDict():
#     result = {}
#     for lang in LANGUAGES:
#         result[lang] = f"./{lang}/SourceMeter{lang}.exe"
#
#
# SOURCE_METER = getSourceMeterDict()
