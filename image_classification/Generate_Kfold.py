from scripts import dataset
from scripts.tools import *
import warnings
import gc
warnings.filterwarnings('ignore')

cls_data = dataset.classification(dataset_root='/path/to/fundus',
                                  normalize=True, polar = True, contrast=True, do_polar_aug=True, polar_augment_count=4,
                                  fliplr=True, shuffle=True, img_size=(256,512), zoom_in=True, search_angle=True, 
                                  equalize=False, sharpening=True, median=True, batch_size = 32, workers = 36, 
                                  OD_only=False, select_channel='g', filename = 'Glaucoma_processed_K1.pkl',
                                  save_info=True, save_batchset=True, split_mode='kfold', num_total_K=10, test_K = '1')

for k in range(2,11):
    k = str(k)
    filename = 'Glaucoma_processed_K{}.pkl'.format(k)
    cls_data.change_K_fold(k, filename=filename, reload=True, save_batchset=True)