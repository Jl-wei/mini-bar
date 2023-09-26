# Calculate the precision, recall, F1 of the result files

import os
import argparse

def print_latex_table_in_one_line(class_value, class_line, i):
    head = '& '
    results = []
    for key in class_line:
        head += key + ' & '
        results.extend(str(x) for x in list(map(lambda x: round(x/i, 3), class_value[key])))
        # text += ' & '.join(str(x) for x in list(map(lambda x: round(x/i, 3), class_value[key])))
    text = ' & '.join(results) + ' \\\\'
    print(head)
    print(text)

def print_latex_table_line(name, lis, i):
    text = '\t' + name + ' & ' + ' & '.join(str(x) for x in list(map(lambda x: round(x/i, 3), lis))) + ' \\\\'
    print(text)

def get_cumulate_value(lines, row_number, lis):
    return [sum(x) for x in zip(lis, list(map(lambda x: float(x), lines[row_number].split()[-4:-1])))]

def analyse(log_file_start_with, print_as, class_line):
    class_value = {}
    for key in class_line:
        class_value[key] = [0,0,0,0]
        # line number - 1 because array index start from 0
        class_line[key] = class_line[key] - 1
    i = 0    
    for dirname in os.listdir('./lightning_logs'):
        root, ext = os.path.splitext(dirname)
        if any(root.startswith(x) for x in log_file_start_with):
            f = open(f"./lightning_logs/{dirname}/log.log", "r")
            lines = f.readlines()
            
            for key in class_line:
                class_value[key] = get_cumulate_value(lines, class_line[key], class_value[key])

            i += 1

    if print_as == 'line':
        print_latex_table_in_one_line(class_value, class_line, i)
    else:
        print(log_file_start_with)
        print("""\\begin{table}[!htb]
        \\centering
        \\begin{tabular}{l c c c}""")
        print("\\hline")
        print("& Precision & Recall & F1 \\\\")
        print("\\hline")
        for key in class_line:
            print_latex_table_line(key, class_value[key], i)
        print("\\hline")
        print(f"""\
        \\end{{tabular}}
        \\caption{{{log_file_start_with}}}
        \\label{{tab:my_label}}""")
        print("\\end{table}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average from multiple log files')
    parser.add_argument('--file', nargs='+', action='store')
    parser.add_argument('--i', type=int, action='store')
    parser.add_argument('--f', type=int, action='store')
    parser.add_argument('--b', type=int, action='store')
    parser.add_argument('--a', type=int, action='store')
    parser.add_argument('--print_as', action='store', default='line')
    args = parser.parse_args()
    
    analyse(args.file, args.print_as, {
        'Feature Request': args.f, 
        'Problem Report': args.b,         
        'Irrelevant': args.i, 
        '\\textbf{Weighted Average}': args.a
    })
    
    # python log_analyser.py --file bert-Garmin-Huawei-Samsung --i 7 --f 8 --b 9 --a 13
    # python log_analyser.py --file camembert-Garmin-Huawei-Samsung --i 7 --f 8 --b 9 --a 13
    # python log_analyser.py --file xlm-roberta-Garmin-Huawei-Samsung --i 7 --f 8 --b 9 --a 13
    # python log_analyser.py --file xlm-roberta-Garmin-Huawei-Samsung --i 22 --f 23 --b 24 --a 28
    