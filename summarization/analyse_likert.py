import pandas as pd
import re
from pathlib import Path


# Analyse the filled likert file to calculate the average score
def analyse_likert(file_path):
    dim = ['Relevance', 'Consistency', 'Fluency', 'Coherence']
    en_result = [0, 0, 0, 0]
    fr_result = [0, 0, 0, 0]
    bi_result = [0, 0, 0, 0]
    en_summary_num = 0
    fr_summary_num = 0
    bi_summary_num = 0
    lang = ''
    with open(file_path, 'r') as f:
        for line in f:
            for i in range(4):
                if re.match('^\# (Garmin Connect|Huawei Health|Samsung Health)\.en', line):
                    lang = 'en'
                elif re.match('^\# (Garmin Connect|Huawei Health|Samsung Health)\.fr', line):
                    lang = 'fr'
                elif re.match('^\# (Garmin Connect|Huawei Health|Samsung Health)\.bi', line):
                    lang = 'bi'

                if re.match('^\| +\*\*{}\*\* +\|'.format(dim[i]), line):
                    if lang == 'en':
                        en_summary_num += 1
                        for index, e in enumerate(line.split('|')):
                            if re.search('X|x', e):
                                en_result[i] += (index-1)
                    elif lang == 'fr':
                        fr_summary_num += 1
                        for index, e in enumerate(line.split('|')):
                            if re.search('X|x', e):
                                fr_result[i] += (index-1)
                    elif lang == 'bi':
                        bi_summary_num += 1
                        for index, e in enumerate(line.split('|')):
                            if re.search('X|x', e):
                                bi_result[i] += (index-1)

    # en_result = list(map(lambda e: round(e/(en_summary_num/4), 2), en_result))
    # fr_result = list(map(lambda e: round(e/(fr_summary_num/4), 2), fr_result))
    bi_result = list(map(lambda e: round(e/(bi_summary_num/4), 2), bi_result))
    
    # print(en_result)
    # print(fr_result)
    print(bi_result)

if __name__ == '__main__':    
    analyse_likert('./likert/likert.md')