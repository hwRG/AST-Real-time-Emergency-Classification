import numpy as np
import json
import os

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]


if not os.path.exists('./data/Emergency_data/wav_16k/'):
    base_dir = './data/Emergency_data/'
    os.mkdir('./data/Emergency_data/wav_16k/')
    cat_list = os.listdir('./data/Emergency_data/wav')
    print(cat_list)
    for cat in cat_list:
        os.mkdir('./data/Emergency_data/wav_16k/' + cat)
        wav_list = os.listdir('./data/Emergency_data/wav/' + cat)
        for wav in wav_list:
            print('sox ' + base_dir + 'wav/' + cat + '/' + wav + ' -r 16000 ' + base_dir + 'wav_16k/' + cat + '/' + wav)
            os.system('sox ' + base_dir + 'wav/' + cat + '/' + wav + ' -r 16000 ' + base_dir + 'wav_16k/' + cat + '/' + wav)


label_set = np.loadtxt('./data/Emergency_class_labels_indices.csv', delimiter=',', dtype='str')
print(label_set)
label_map = {}
for i in range(1, len(label_set)):
    print(label_set[i][2])
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

# fix bug: generate an empty directory to save json files
if os.path.exists('./data/datafiles_16k') == False:
    os.mkdir('./data/datafiles_16k')

for fold in [1,2,3,4,5]:
    base_path = "./data/Emergency_data/wav_16k/"
    meta = np.loadtxt('./data/Emergency_data/Emergency.csv', delimiter=',', dtype='str', skiprows=1)
    train_wav_list = []
    eval_wav_list = []
    for i in range(0, len(meta)):
        cur_label = label_map[meta[i][3]]
        cur_path = meta[i][0]
        cur_fold = int(meta[i][1])
        # /m/07rwj is just a dummy prefix <= esc 레이블인듯
        cur_dict = {"wav": base_path + cur_path, "labels": '/m/09xr'+cur_label.zfill(2)} # !! 레이블 이름 지정 09xr로 통일
        if cur_fold == fold:
            eval_wav_list.append(cur_dict)
        else:
            train_wav_list.append(cur_dict)

    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

    # !!! 유니코드로 나오는 문제를 ensure_ascii=False로 해결
    with open('./data/datafiles_16k/Emergency_train_data_'+ str(fold) +'.json', 'w', encoding='UTF-8-sig') as f:
        json.dump({'data': train_wav_list}, f, indent=1, ensure_ascii=False)

    with open('./data/datafiles_16k/Emergency_eval_data_'+ str(fold) +'.json', 'w', encoding='UTF-8-sig') as f:
        json.dump({'data': eval_wav_list}, f, indent=1, ensure_ascii=False)

print('Finished Emergency Preparation')