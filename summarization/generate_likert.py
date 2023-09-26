import pandas as pd
import re
from pathlib import Path

# Generate a likert file of the summaries
def generate_likert(dir_path):
    Path("./likert").mkdir(parents=True, exist_ok=True)
    f = open("./likert/likert.md", "w")
    
    path = Path(__file__).parent.joinpath(dir_path).resolve()
    for app_folder in path.iterdir():
        if app_folder.is_file(): continue
        if app_folder.suffix == '':
            f.write('# {}.bi\n'.format(app_folder.name))
        else:
            f.write('# {}\n'.format(app_folder.name))
        for file in app_folder.iterdir():
            f.write('## {}\n'.format(file.stem))
            if not file.suffix == '.csv': continue
            df = pd.read_csv(file)
            for index, row in df.iterrows():
                f.write('### {}\n'.format(index))
                f.write('**User reviews:**\n')
                for doc in row['document'].split('|||||'):
                    f.write('* {}\n'.format(doc))
                f.write("\n")
                f.write('**Summary:**\n')
                f.write('{}\n'.format(row['summary']))

                f.write('''
|                 | Very unsatisfied | Unsatisfied | Neutral | Satisfied | Very satisfied |
| --------------- | ---------------- | ----------- | ------- | --------- | -------------- |
| **Relevance**   |                  |             |         |           |                |
| **Consistency** |                  |             |         |           |                |
| **Fluency**     |                  |             |         |           |                |
| **Coherence**   |                  |             |         |           |                |
                ''')
                f.write("\n")
            f.write("\n")
        f.write("\n")
    f.close()
    
if __name__ == '__main__':    
    generate_likert("./output/")