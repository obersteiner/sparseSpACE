from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import re

def prev_level(l, d):
    if l - 2 <= 0:
        return 1
    else:
        return (2**(l-2) - 1) * d + prev_level(l-2, d)

with open('log_sg') as f:
    content = f.readlines()
content = [x.split('::')[1] for x in content]
content = [x.strip() for x in content]
filterDelimiters = lambda x: not ('|||' in x or '~~~' in x or '###' in x or '---' in x)
content = [x for x in content if filterDelimiters(x)]
data_set_indices = [i for i, x in enumerate(content) if 'do.datasets' in x]
data_sets = [content[data_set_indices[x]:data_set_indices[x+1]] for x in range(len(data_set_indices)) if (x+1) < len(data_set_indices)]
substr_idx = lambda s, sub: [i for i, x in enumerate(s) if sub in x]
stringIndex = lambda l, st: [i for i, x in enumerate(l) if st in x][0]
#test = [s for s in data_sets for i in scheme_indices(s)]
#test2 = [substr_idx(s, 'Percentage') for s in data_sets]
data_sets = [(s[:stringIndex(s, 'data dimension')+1], s[stringIndex(s, 'data dimension')+1:substr_idx(s, 'Percentage')[0]+1], s[substr_idx(s, 'Percentage')[0]+1:]) for s in data_sets]

eval_avg = {}
for tup in data_sets:
    meta = tup[0]
    data_size = int(meta[substr_idx(meta, 'data size')[0]].split(':')[1])
    data_dim = int(meta[substr_idx(meta, 'data dimension')[0]].split(':')[1])
    data_type = meta[substr_idx(meta, 'make_')[0]]

    stdCombi = tup[1]
    stdTime = [float(s.split(':')[1]) for s in stdCombi if 'Time' in s]
    max_level_std = [int(s.split(':')[1]) for s in stdCombi if 'max_level' in s][0]
    point_num = ((2**max_level_std) - 1) * data_dim - (data_dim - 1) + (2**data_dim) * prev_level(max_level_std, data_dim)
    #stdPoints = [((2 ** int(s.split(':')[1])) - 1) * data_dim for s in stdCombi if 'max_level' in s]
    stdPoints = [point_num]
    stdAcc = [float(s.split(':')[1].replace('%', '')) for s in stdCombi if 'Percentage' in s]

    dimCombi = tup[2]
    dimTime = [float(s.split(':')[1]) for s in dimCombi if 'Time' in s]
    dimPoints = [int(s.split(':')[1]) for s in dimCombi if 'distinct points' in s]
    #dimPoints = [int(s.split(':')[1]) for s in dimCombi if 'max_evaluations' in s]
    dimAcc = [float(s.split(':')[1].replace('%', '')) for s in dimCombi if 'Percentage' in s]

    if data_type not in eval_avg.keys():
        eval_avg[data_type] = {}
    std_info = namedtuple('std_info', 'stdTimeAvg stdPointAvg stdAcc')
    dim_info = namedtuple('dim_info', 'dimTimeAvg dimPointAvg dimAcc')
    entry = (std_info(sum(stdTime)/len(stdTime), sum(stdPoints)/len(stdPoints), stdAcc),
             dim_info(sum(dimTime)/len(dimTime), sum(dimPoints)/len(dimPoints), dimAcc))
    if data_dim not in eval_avg[data_type].keys():
        eval_avg[data_type][data_dim] = []
    eval_avg[data_type][data_dim].append(entry)


# plot the data
for dtype in eval_avg.keys():
    for dimension in eval_avg[dtype].keys():
        regex = r"make_.*\("
        match = re.search(regex, str(dtype))
        name = match.group(0)
        name = name.replace('make_', '').replace('(', '')

        entries = eval_avg[dtype][dimension]
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

        plt.figure(num=3, figsize=(8, 5))
        plt.plot(time_std_x, acc_y_std, label='StdCombi')
        plt.plot(time_dim_x, acc_y_dim, label='DimCombi', color='red', linewidth=1.0)
        plt.xlabel("Runtime")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper right")
        plt.title(name + '\ndimensions: ' + str(dimension))
        plt.savefig('eval_figs/'+name+'_'+str(dimension)+'_time_acc', bbox_inches='tight')
        plt.show()

        # plot points and accuracy
        points_std_x = np.linspace(min(stdPoints), max(stdPoints), len(stdPoints))
        points_dim_x = np.linspace(min(dimPoints), max(dimPoints), len(dimPoints))
        plt.figure(num=3, figsize=(8, 5))
        plt.plot(points_std_x, acc_y_std, label='StdCombi')
        plt.plot(points_dim_x, acc_y_dim, label='DimCombi', color='red', linewidth=1.0)
        plt.xlabel("# Points")
        plt.ylabel("Accuracy")
        plt.title(name + '\ndimensions: ' + str(dimension))
        plt.legend(loc="upper right")
        plt.savefig('eval_figs/'+name+'_'+str(dimension)+'_points_acc', bbox_inches='tight')
        plt.show()

        # plot runtime and points
        point_y_std = np.array(stdPoints).flatten()
        point_y_dim = np.array(dimPoints).flatten()
        plt.figure(num=3, figsize=(8, 5))
        plt.plot(time_std_x, point_y_std, label='StdCombi')
        plt.plot(time_dim_x, point_y_dim, label='DimCombi', color='red', linewidth=1.0)
        plt.xlabel("Runtime")
        plt.ylabel("# Points")
        plt.title(name + '\ndimensions: ' + str(dimension))
        plt.legend(loc="upper right")
        plt.savefig('eval_figs/'+name+'_'+str(dimension)+'_time_points', bbox_inches='tight')
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
