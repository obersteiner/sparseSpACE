from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import re

def prev_level(l, d):
    if l - 2 <= 0:
        return 1
    else:
        return (2**(l-2) - 1) * d + prev_level(l-2, d)

with open('log_sg_real-data_backup_2') as f:
    content = f.readlines()
content = [x.split('::')[1] for x in content]
content = [x.strip() for x in content]
filterDelimiters = lambda x: not ('|||' in x or '~~~' in x or '###' in x or '---' in x)
content = [x for x in content if filterDelimiters(x)]
data_set_indices = [i for i, x in enumerate(content) if 'next iteration' in x]
if True in ['do.datasets' in x for x in content]:
    data_set_indices = [i for i, x in enumerate(content) if 'do.datasets' in x]
elif True in ['data set' in x for x in content]:
    data_set_indices = [i for i, x in enumerate(content) if 'data set' in x]
elif True in ['next iteration' in x for x in content]:
    data_set_indices = [i for i, x in enumerate(content) if 'next iteration' in x]
else:
    print('couldn\'t parse log')
data_sets = [content[data_set_indices[x]:data_set_indices[x+1]] for x in range(len(data_set_indices)) if (x+1) < len(data_set_indices)]

substr_idx = lambda s, sub: [i for i, x in enumerate(s) if sub in x]
stringIndex = lambda l, st: [i for i, x in enumerate(l) if st in x][0]

#data_sets = [(s[:stringIndex(s, 'data dimension')+1], s[stringIndex(s, 'data dimension')+1:substr_idx(s, 'Percentage')[0]+1], s[substr_idx(s, 'Percentage')[0]+1:]) for s in data_sets]
data_sets = [(s[:stringIndex(s, 'one_vs_others')+1], s[stringIndex(s, 'one_vs_others')+1:substr_idx(s, 'Percentage')[0]+1], s[substr_idx(s, 'Percentage')[0]+1:]) for s in data_sets]


eval_avg = {}
for tup in data_sets:
    str_exists = lambda lines, substr: len(substr_idx(lines, substr)) > 0
    meta = tup[0]
    data_size = int(meta[substr_idx(meta, 'data size')[0]].split(':')[1]) if str_exists(meta, 'data size') else None
    data_dim = int(meta[substr_idx(meta, 'data dimension')[0]].split(':')[1]) if str_exists(meta, 'data dimension') else None
    data_type = meta[substr_idx(meta, 'make_')[0]] if str_exists(meta, 'make_') else meta[0]
    one_vs_others = meta[substr_idx(meta, 'one_vs_others')[0]].split()[1] if str_exists(meta, 'one_vs_others') else None
    error_calculator = meta[substr_idx(meta, 'error_calculator')[0]].split()[-1] if str_exists(meta, 'error_calculator') else None
    rebalancing = meta[substr_idx(meta, 'rebalancing')[0]].split()[-1] if str_exists(meta, 'rebalancing') else None
    margin = meta[substr_idx(meta, 'margin')[0]].split()[-1] if str_exists(meta, 'margin') else None

    stdCombi = tup[1]
    stdTime = [float(s.split(':')[1]) for s in stdCombi if 'Time' in s] if str_exists(stdCombi, 'Time') else None
    max_level_std = [int(s.split(':')[1]) for s in stdCombi if 'max_level' in s][0] if str_exists(stdCombi, 'max_level') else None
    point_num = ((2**max_level_std) - 1) * data_dim - (data_dim - 1) + (2**data_dim) * prev_level(max_level_std, data_dim) if max_level_std is not None else 0
    #stdPoints = [((2 ** int(s.split(':')[1])) - 1) * data_dim for s in stdCombi if 'max_level' in s]
    stdPoints = [point_num*len(stdTime)]
    stdAcc = [float(s.split(':')[1].replace('%', '')) for s in stdCombi if 'Percentage' in s] if str_exists(stdCombi, 'Percentage') else None

    dimCombi = tup[2]
    dimTime = [float(s.split(':')[1]) for s in dimCombi if 'Time used adaptive' in s] if str_exists(dimCombi, 'Time') else None
    dimPoints = [int(s.split(':')[1]) for s in dimCombi if 'distinct points' in s] if str_exists(dimCombi, 'distinct points') else None
    dimStartLevel = [int(s.split(':')[1]) for s in dimCombi if 'dimwise start level' in s] if str_exists(dimCombi, 'dimwise start level') else None
    #dimPoints = [int(s.split(':')[1]) for s in dimCombi if 'max_evaluations' in s]
    dimAcc = [float(s.split(':')[1].replace('%', '')) for s in dimCombi if 'Percentage' in s] if str_exists(dimCombi, 'Percentage') else None

    if data_type not in eval_avg.keys():
        eval_avg[data_type] = {}
    std_info = namedtuple('std_info', 'stdTimeAvg stdPointAvg stdAcc')
    dim_info = namedtuple('dim_info', 'dimTimeAvg dimPointAvg dimAcc')
    entry = (std_info(sum(stdTime)/len(stdTime), sum(stdPoints)/len(stdPoints), stdAcc, ),
             dim_info(sum(dimTime)/len(dimTime), sum(dimPoints)/len(dimPoints), dimAcc))

    #if old log file type
    # key = data_dim
    # else
    if rebalancing is not None and margin is not None:
        key = str([data_dim, dimStartLevel, one_vs_others, error_calculator, margin, rebalancing])
    else:
        key = str([data_dim, dimStartLevel, one_vs_others, error_calculator])
    if key not in eval_avg[data_type].keys():
        eval_avg[data_type][key] = []
    eval_avg[data_type][key].append(entry)


# plot the data
for dtype in eval_avg.keys():
    #for dimension in eval_avg[dtype].keys():
    for keys in eval_avg[dtype].keys():
        if isinstance(keys, int):
            dimension = keys
            dimStartLevel_str = ''
            ovo_str = ''
            error_calculator = ''
        else:
            k = str(keys).replace('[', '').replace(']', '').replace('\'','')
            k = k.split(',')
            dimension = int(k[0])
            dimStartLevel = int(k[1])
            dimStartLevel_str = '_DimWiseStartLvl_'+str(dimStartLevel)
            one_vs_others = k[2]
            ovo_str = '_one_vs_others' if 'True' in one_vs_others else ''
            error_calculator = k[3].split('.')[1]
            margin = ''
            if len(k) > 4:
                margin = 'margin-' + str(k[4]).replace('margin', '').replace(':', '').strip().replace('.', 'p')
            rebalancing = ''
            if len(k) > 5:
                rebalancing = 'rebalancing' if 'True' in str(k[5]) else ''


        regex = r"make_.*\("
        match = re.search(regex, str(dtype))
        if match is not None:
            name = match.group(0)
            name = name.replace('make_', '').replace('(', '')
        else:
            name = dtype.replace(' ', '_')

        entries = eval_avg[dtype][keys]
        stdTimes = [t[0].stdTimeAvg for t in entries]
        stdPoints = [t[0].stdPointAvg for t in entries]
        stdAccs = [t[0].stdAcc for t in entries]

        dimTimes = [t[1].dimTimeAvg for t in entries]
        dimPoints = [t[1].dimPointAvg for t in entries]
        dimAccs = [t[1].dimAcc for t in entries]

        # plot time and accuracy
        time_std_x = np.linspace(min(stdTimes), max(stdTimes), len(stdTimes))
        time_dim_x = np.linspace(min(dimTimes), max(dimTimes), len(dimTimes))
        acc_y_std = np.array(stdAccs).flatten()
        acc_y_dim = np.array(dimAccs).flatten()
        min_acc = min(np.min(acc_y_std), np.min(acc_y_dim))

        plt.figure(num=3, figsize=(16, 10))
        bar_width = 3.0
        line_width = 2.0
        if len(time_std_x) == 1 and len(time_dim_x) == 1:
            plt.bar([time_std_x[0]], [acc_y_std[0]], label='standard combi', width=bar_width)
            plt.bar([time_dim_x[0]], [acc_y_dim[0]], label='dimwise combi', width=bar_width)
            plt.legend(loc="upper center")
        else:
            plt.plot(time_std_x, acc_y_std, label='StdCombi', linewidth=line_width)
            plt.plot(time_dim_x, acc_y_dim, label='DimCombi', color='red', linewidth=line_width)
            plt.ylim(bottom=min(50.0, min_acc), top=100.0)
            plt.legend(loc="upper right")
        plt.xlabel("Runtime")
        plt.ylabel("Accuracy")
        plt.title(name + '\ndimensions: ' + str(dimension) + '\ndimWise start level : ' + str(dimStartLevel) + '\n'+ovo_str + '\nerror_calculator: '+error_calculator + '\nmargin: '+margin.replace('p', '.')+'\nrebalancing: '+('True' if rebalancing is not '' else 'False'))
        plt.savefig('eval_figs/time_acc/'+name+'_'+str(dimension)+'_time_acc'+dimStartLevel_str+ovo_str+'_'+error_calculator+'_'+margin+'_'+rebalancing, bbox_inches='tight')
        plt.show()

        # plot points and accuracy
        points_std_x = np.linspace(min(stdPoints), max(stdPoints), len(stdPoints))
        points_dim_x = np.linspace(min(dimPoints), max(dimPoints), len(dimPoints))
        plt.figure(num=3, figsize=(16, 10))
        if len(points_std_x) == 1 and len(points_dim_x) == 1:
            plt.bar([points_std_x[0]], [acc_y_std[0]], label='standard combi', width=bar_width)
            plt.bar([points_dim_x[0]], [acc_y_dim[0]], label='dimwise combi', width=bar_width)
            plt.legend(loc="upper center")
        else:
            plt.plot(points_std_x, acc_y_std, label='StdCombi', linewidth=line_width)
            plt.plot(points_dim_x, acc_y_dim, label='DimCombi', color='red', linewidth=line_width)
            plt.ylim(bottom=min(50.0, min_acc), top=100.0)
            plt.legend(loc="upper right")
        plt.xlabel("# Points")
        plt.ylabel("Accuracy")
        plt.title(name + '\ndimensions: ' + str(dimension) + '\ndimWise start level : ' + str(dimStartLevel) + '\n'+ovo_str + '\nerror_calculator: '+error_calculator + '\nmargin: '+margin.replace('p', '.')+'\nrebalancing: '+('True' if rebalancing is not '' else 'False'))
        plt.savefig('eval_figs/point_acc/'+name+'_'+str(dimension)+'_points_acc'+dimStartLevel_str+ovo_str+'_'+error_calculator+'_'+margin+'_'+rebalancing, bbox_inches='tight')
        plt.show()

        # plot runtime and points
        point_y_std = np.array(stdPoints).flatten()
        point_y_dim = np.array(dimPoints).flatten()
        plt.figure(num=3, figsize=(16, 10))
        if len(time_std_x) == 1 and len(time_dim_x) == 1:
            plt.bar([time_std_x[0]], [point_y_dim[0]], label='standard combi', width=bar_width)
            plt.bar([time_dim_x[0]], [point_y_dim[0]], label='dimwise combi', width=bar_width)
            plt.legend(loc="upper center")
        else:
            plt.plot(time_std_x, point_y_std, label='StdCombi', linewidth=line_width)
            plt.plot(time_dim_x, point_y_dim, label='DimCombi', color='red', linewidth=line_width)
            plt.legend(loc="upper right")
        plt.xlabel("Runtime")
        plt.ylabel("# Points")
        plt.title(name + '\ndimensions: ' + str(dimension) + '\ndimWise start level : ' + str(dimStartLevel) + '\n'+ovo_str + '\nerror_calculator: '+error_calculator + '\nmargin: '+margin.replace('p', '.')+'\nrebalancing: '+('True' if rebalancing is not '' else 'False'))
        plt.savefig('eval_figs/time_point/'+name+'_'+str(dimension)+'_time_points'+dimStartLevel_str+ovo_str+'_'+error_calculator+'_'+margin+'_'+rebalancing, bbox_inches='tight')
        plt.show()




print('end')


# with open('log_old') as f:
#     content = f.readlines()
# content = [x.split('::')[1] for x in content]
# content = [x.strip() for x in content]
# filterDelimiters = lambda x: not ('|||' in x or '~~~' in x or '###' in x or '---' in x)
# content = [x for x in content if filterDelimiters(x)]
# data_set_indices = [i for i, x in enumerate(content) if 'data size' in x]
# data_sets = [content[data_set_indices[x]:data_set_indices[x+1]] for x in range(len(data_set_indices)) if (x+1) < len(data_set_indices)]
# substr_idx = lambda s, sub: [i for i, x in enumerate(s) if sub in x]
# stringIndex = lambda l, st: [i for i, x in enumerate(l) if st in x][0]
# #test = [s for s in data_sets for i in scheme_indices(s)]
# #test2 = [substr_idx(s, 'Percentage') for s in data_sets]
# data_sets = [(s[:stringIndex(s, 'datasets')+1], s[stringIndex(s, 'datasets')+1:substr_idx(s, 'Percentage')[0]+1], s[substr_idx(s, 'Percentage')[0]+1:]) for s in data_sets]
#
# eval_avg = {}
# for tup in data_sets:
#     meta = tup[0]
#     data_size = int(meta[substr_idx(meta, 'data size')[0]].split(':')[1])
#     data_dim = int(meta[substr_idx(meta, 'data dimension')[0]].split(':')[1])
#     data_type = meta[substr_idx(meta, 'make_')[0]]
#
#     stdCombi = tup[1]
#     stdTime = [float(s.split(':')[1]) for s in stdCombi if 'Time' in s]
#     max_level_std = [int(s.split(':')[1]) for s in stdCombi if 'max_level' in s][0]
#     point_num = ((2**max_level_std) - 1) * data_dim - (data_dim - 1) + (2**data_dim) * prev_level(max_level_std, data_dim)
#     #stdPoints = [((2 ** int(s.split(':')[1])) - 1) * data_dim for s in stdCombi if 'max_level' in s]
#     stdPoints = [point_num]
#     stdAcc = [float(s.split(':')[1].replace('%', '')) for s in stdCombi if 'Percentage' in s]
#
#     dimCombi = tup[2]
#     dimTime = [float(s.split(':')[1]) for s in dimCombi if 'Time' in s]
#     dimPoints = [int(s.split(':')[1]) for s in dimCombi if 'distinct points' in s]
#     dimAcc = [float(s.split(':')[1].replace('%', '')) for s in dimCombi if 'Percentage' in s]
#
#     if data_type not in eval_avg.keys():
#         eval_avg[data_type] = {}
#     std_info = namedtuple('std_info', 'stdTimeAvg stdPointAvg stdAcc')
#     dim_info = namedtuple('dim_info', 'dimTimeAvg dimPointAvg dimAcc')
#     entry = (std_info(sum(stdTime)/len(stdTime), sum(stdPoints)/len(stdPoints), stdAcc),
#              dim_info(sum(dimTime)/len(dimTime), sum(dimPoints)/len(dimPoints), dimAcc))
#     if data_dim not in eval_avg[data_type].keys():
#         eval_avg[data_type][data_dim] = []
#     eval_avg[data_type][data_dim].append(entry)
