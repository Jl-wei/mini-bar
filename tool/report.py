import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from utilities import read_json

if __name__ == '__main__':
    environment = Environment(loader=FileSystemLoader("./"))
    template = environment.get_template("report_template.txt")

    path = Path(__file__).parent.joinpath('./reports').resolve()
    reports = []
    for report_path in path.iterdir():
        report = read_json(report_path)
        report['app'] = '-'.join(report['app'].split(' '))
        reports.append(report)

    content = template.render(reports=reports)

    with open('report.html', mode="w", encoding="utf-8") as f:
        f.write(content)