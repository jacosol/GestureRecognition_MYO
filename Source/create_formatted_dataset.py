from DSformatting import *
import torch.nn.functional as F

import os

path = r'C:\Users\Copo\source\\repos\GestureRecognition\GestureRecognition_MYO\EMG_data_for_gestures-master\EMG_data_for_gestures-master'
save_root = r'C:\Users\Copo\source\\repos\GestureRecognition\GestureRecognition_MYO\CustomDataset\\txts'

create_dataset_dirs(save_root)
# get all the txt files in the raw data
files = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*.txt'))]
# runs over all files and saves each processed sample to create the custom dataset
k = 1
for file in files[1:]:
    res = create_single_txt_sample(file, k, save_root)
    if res:
        k = k+2


raw_ds_path = r'C:\Users\Copo\source\\repos\GestureRecognition\GestureRecognition_MYO\CustomDataset\\txts'
processed_ds_path = r'C:\Users\Copo\source\\repos\GestureRecognition\GestureRecognition_MYO\CustomDataset\processed'
# create folders to store training samples (data and labels) for each label
create_dataset_dirs(processed_ds_path)
print(glob.glob(os.path.join(processed_ds_path,'*')))

# loads the txts for each channel, gets the label, create stacks of 8 MYO channels and corresponding label
process_data_set(raw_ds_path, processed_ds_path)  