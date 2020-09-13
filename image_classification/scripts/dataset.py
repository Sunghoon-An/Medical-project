# -*- coding: utf-8 -*-

from scripts.tools import *
from scripts.preprocess import *
from tqdm import tqdm
from scipy.ndimage import rotate
from skimage import *
from copy import deepcopy
from sklearn.model_selection import ShuffleSplit

import numpy as np
import gc, pickle
import datetime

from multiprocessing import Pool

class classification:
    '''
    Description :
        Classification을 하기 위해 정상 안저와 녹내장 안저영상을 전처리하기 위한 Class 정의
    
    Attributes :  ## V : 변수 , F : 함수 ##
        - [V] dataset_root : 데이터세트 경로
        - [V] img_size : 사용할 이미지 사이즈
        - [V] normalize : 전처리 함수 'normalize_img' 적용 여부
        - [V] polar : 전처리 함수 'polar_transform' 적용 여부
        - [V] do_polar_aug : 전처리 함수 'polar_transform'를 사용하여 Augmentation 수행 여부
        - [V] polar_augment_count : 전처리 함수 'polar_transform'를 사용한 Augmentation을 얼마나 수행할 지 명시
        - [V] search_angle : 전처리 함수 'polar_transform' 적용시에 최적 각도 탐색 여부 
        - [V] contrast : 영상 색채 강화 여부
        - [V] rotation : 영상 회전 여부
        - [V] rotation_theta_interval : 영상 회전 시에 적용할 각도 간격
        - [V] fliplr : 영상 미러링 여부
        - [V] shuffle : 데이터세트 셔플 여부
        - [V] split_ratio : 데이터셋 분할 비율 (유효 데이터세트 비율)
        - [V] workers : 병렬처리 할 때 사용할 Thread 수
        - [V] equalize : 전처리 함수 'equalize_img' 적용 여부
        - [V] sharpening : 전처리 함수 'sharpening_filter' 적용 여부
        - [V] median : 전처리 함수 'median_filter' 적용 여부
        - [V] zoom_in : 전처리 함수 'zoomIN' 적용 여부
        - [V] select_channel : 색채널 사용 정도 {'RGB', 'R', 'G', 'B'}
        - [V] split_mode : 데이터세트 분할 방법
        - [V] batch_size : batch size
        - [V] random_seed : Kfold로 분할 및 데이터세트 셔플을 할 경우 사용할 Random seed 명시
        - [V] num_total_K : 분할 모드가 Kfold 일 때 사용한 K 수
        - [V] test_K : 분할 모드가 Kfold 일 때 현재 사용중인 K번째
        - [V] Kfoldset : K개 만큼으로 쪼개진 Train, Validation dataset 경로 모음.
        
        - [F] __attr2dict : 현재 설정된 전처리 옵션들(Class Attributes)을 dictionary 형태로 변환
        - [F] __process_data : 정해진 순서에 따라 주어진 데이터를 전처리 수행
        - [F] _pipeline : Augmentation 등 하나의 데이터를 처리하기위해 정의된 모든 함수
        - [F] _parallelized_load : '_pipeline'을 병렬처리하기 위한 함수
        - [F] print_settings : 현재 설정되어있는 전처리 정보들을 출력
        - [F] change_batch_size : 현재 설정되어 있는 batch_size를 주어진 batch_size로 재분할
        - [F] repreprocess : 현재 데이터세트를 Class에서 설정되어있는 Attributes에 따라 재전처리 수행
        - [F] change_K_fold : Kfold를 수행하여 처리하여 미리 저장된 데이터 세트를 주어진 K에 따라 데이터세트 재처리
    '''
    
    def __init__(self, dataset_root='./dataset/',
                 img_size=(256,256), normalize=False, polar = False, do_polar_aug=False, polar_augment_count=1, 
                 search_angle=True, contrast = False, fliplr=False, rotation=False, rotation_theta_interval = 20, 
                 shuffle=False, equalize=False, select_channel='rgb', sharpening = False, median = False,
                 batch_size = None, split_ratio=0.8, workers=4, zoom_in=False, 
                 filename = 'GD_phase_division.pkl', save_info=False, save_batchset=False,
                 split_mode='random', random_seed=20141031, num_total_K=5, test_K = '1', restore=False):
        '''
        Description :
            Classification 용 데이터 세트를 주어진 전처리 옵션에 따라 안저영상을 전처리하여 객체의 attributes로 저장.
            혹은 전처리 되어 저장해둔 전처리 결과 파일을 불러와서 사용할 수 있으며, 옵션을 바꾸어 재처리도 가능.
            데이터 분할 시에 Kfold로 분할 하였다면 K셋 또한 변경할 수 있음. 
            
        Arguments : 
            - dataset_root (기본값 : ./dataset) : 데이터세트 경로
            - img_size (기본값 : (256,256)) : 사용할 이미지 사이즈
            - normalize (기본값 : False) : 전처리 함수 'normalize_img' 적용 여부
            - polar (기본값 : False) : 전처리 함수 'polar_transform' 적용 여부
            - do_polar_aug (기본값 : False) : 전처리 함수 'polar_transform'를 사용하여 Augmentation 수행 여부
            - polar_augment_count (기본값 : 1) : 전처리 함수 'polar_transform'를 사용한 Augmentation을 얼마나 수행할 지 명시
            - search_angle (기본값 : True) : 전처리 함수 'polar_transform' 적용시에 최적 각도 탐색 여부 
            - contrast (기본값 : False) : 영상 색채 강화 여부
            - fliplr (기본값 : False) : 영상 미러링 여부
            - rotation (기본값 : False) : 영상 회전 여부
            - rotation_theta_interval (기본값 : 20) : 영상 회전 시에 적용할 각도 간격
            - shuffle : 데이터세트 셔플 여부
            - equalize (기본값 : False) : 전처리 함수 'equalize_img' 적용 여부
            - select_channel (기본값 : False) : 색채널 사용 정도 {'RGB', 'R', 'G', 'B'}
            - sharpening (기본값 : False) : 전처리 함수 'sharpening_filter' 적용 여부
            - median (기본값 : False) : 전처리 함수 'median_filter' 적용 여부
            - batch_size (기본값 : None) : batch size
            - split_ratio (기본값 : 0.8) : 데이터셋 분할 비율 (유효 데이터세트 비율)
            - workers (기본값 : 4) : 병렬처리 할 때 사용할 Thread 수
            - zoom_in (기본값 : False) : 전처리 함수 'zoomIN' 적용 여부
            - filename (기본값 : GD_phase_division.pkl) : 전처리한 데이터를 저장할 파일의 이름
            - save_info (기본값 : False) : 전처리 당시 설정하였던 arguments들을 파일과 함께 저장여부
            - save_batchset (기본값 : False) : 전처리된 데이터 저장 여부
            - split_mode (기본값 : False) : 데이터세트 분할 방법
            - random_seed (기본값 : 20141031) : Kfold로 분할 및 데이터세트 셔플을 할 경우 사용할 Random seed 명시
            - num_total_K (기본값 : 5) : 분할 모드가 Kfold 일 때 사용한 K 수
            - test_K (기본값 : 1) : 분할 모드가 Kfold 일 때 현재 사용중인 K번째
            - restore (기본값 : False) : 전처리한 데이터를 다시 불러옴, dataset_root에 전처리된 pkl 파일의 경로를 명시해야함.
        '''
        
        dataset_root += '/' if dataset_root[-1] != '/' and not restore else ''

        self.__label_list = ['normal', 'Glaucoma']
        self.__phase_list = ['train', 'valid']
        self.__data_type = ['image', 'label']
        self.__rgb_list = ['r','g','b']
        
        # 만약 전처리 데이터 호출시, 설정해두었던 전처리 옵션들과 전처리 데이터를 복구
        if restore :
            info = read_pickle(dataset_root)
            self.path_list = info['path_list']
            attr = info['attr']
            
            self.img_size = attr['img_size']
            self.normalize = attr['normalize']
            self.polar = attr['polar']
            self.do_polar_aug = attr['do_polar_aug']
            self.polar_augment_count = attr['polar_augment_count']
            self.search_angle = attr['search_angle']
            self.contrast = attr['contrast']
            self.rotation = attr['rotation']
            self.rotation_theta_interval = attr['rotation_theta_interval']
            self.fliplr = attr['fliplr']
            self.shuffle = attr['shuffle']
            self.split_ratio = attr['split_ratio']
            self.workers = workers
            self.equalize = attr['equalize']
            self.sharpening = attr['sharpening']
            self.median = attr['median']
            self.zoom_in = attr['zoom_in']
            self.select_channel = attr['select_channel']
            self.split_mode = attr['split_mode']

            self.batch_size = attr['batch_size']
            self.random_seed = attr['random_seed']
            self.num_total_K = attr['num_total_K']
            self.test_K = attr['test_K']
            
            self.rand_generator = np.random.RandomState(seed=self.random_seed)
            
            # 전처리 데이터 세트도 같이 저장하였다면 Kfold 복구
            if 'Kfoldset' in info.keys() : 
                self.Kfoldset = info['Kfoldset']
            
            # batch_set으로 쪼개두어서 데이터를 저장해두었으면 해당 데이터 복구
            if 'batch_set' in info.keys() : 
                self.batch_set = info['batch_set']
                
            del info
            gc.collect()
            
        
        # 새롭게 데이터를 전처리 진행
        else :
            self.img_size = img_size
            self.normalize = normalize
            self.polar = polar
            self.do_polar_aug = do_polar_aug
            self.polar_augment_count = polar_augment_count
            self.search_angle = search_angle
            self.contrast = contrast
            self.rotation = rotation
            self.rotation_theta_interval = rotation_theta_interval
            self.fliplr = fliplr
            self.shuffle = shuffle
            self.split_ratio = split_ratio
            self.workers = workers
            self.equalize = equalize
            self.sharpening = sharpening
            self.median = median
            self.zoom_in = zoom_in
            self.select_channel = select_channel
            self.split_mode = split_mode.lower()

            self.batch_size = batch_size
            self.num_total_K = num_total_K
            self.random_seed = random_seed
            self.test_K = test_K
            self.filename = filename
            
            self.rand_generator = np.random.RandomState(seed=self.random_seed)

            if self.split_mode not in ['random', 'kfold'] : 
                raise ValueError("split_mode should be given as None or 'Kfold:K-fold classification'.")
            if int(self.test_K) not in range(1, self.num_total_K+1) :
                raise ValueError("'test_K' cannot be over the 'num_total_K'.")

            self.path_list = {label : [] for label in self.__label_list}
            dataset_path = {phase : [] for phase in self.__phase_list}

            # dataset_root 경로로 부터 이미지 경로들을 녹내장과 정상을 구분하여 수집
            for dirpath, _, filenames in os.walk(dataset_root):
                if 'images' in dirpath :
                    for path in filenames:
                        path = '/'.join([dirpath,path])
                        self.path_list['normal' if 'normal' in path else 'Glaucoma'].append(path)

            # 전체 데이터 세트 셔플
            if self.shuffle:
                for label in self.__label_list :
                    indices = self.rand_generator.permutation(len(self.path_list[label]))
                    self.path_list[label] = list(np.array(self.path_list[label])[indices])

            # 만약 전체 데이터 세트를 'Random'으로 분할할 시에, 주어진 'split_ratio'에 따라
            # 훈련 데이터세트와 유효 데이터세트를 분할
            if self.split_mode == 'random':
                train_ratio = round(len(self.path_list['Glaucoma'])*split_ratio)
                for phase in self.__phase_list: 
                    for label in self.__label_list:
                        dataset_path[phase] += self.path_list[label][:train_ratio] if phase == 'train' else self.path_list[label][train_ratio:]

            # 만약 'Kfold' 형식으로 데이터 세트 분할시 주어진 'num_total_K'만큼 K-fold Cross validation을 수행하기 위해
            # 데이터 세트 경로를 분할하고, 10개의 분할세트를 저장
            elif self.split_mode == 'kfold':
                # 이름 변경
                self.filename = 'Glaucoma_processed_K{}.pkl'.format(self.test_K)
                tmp_list = {phase : [] for phase in self.__phase_list}
                self.Kfoldset = {str(idx+1) : deepcopy(tmp_list) for idx in range(self.num_total_K)}
                
                if type(self.test_K) != str : 
                    self.test_K = str(self.test_K)
                    
                # Label별로 10개씩 분할 후 'train'과 'valid'로 나누어서 재저장.
                for label in self.__label_list:
                    Kfold_idx = ShuffleSplit(n_splits=self.num_total_K, test_size=1/self.num_total_K, random_state = self.random_seed).split(self.path_list[label])
                    Kfold_idx = [kfold for kfold in Kfold_idx]
                    
                    for K, (train_idx, valid_idx) in enumerate(Kfold_idx):
                        K = str(K+1)

                        for idx in train_idx:
                            self.Kfoldset[K]['train'].append(self.path_list[label][idx])
                        for idx in valid_idx:
                            self.Kfoldset[K]['valid'].append(self.path_list[label][idx])
                            
                # Fold별로 학습 세트와 평가 세트를 셔플
                if self.shuffle:
                    for K in range(num_total_K):
                        K = str(K+1)
                        for phase in self.__phase_list:
                            indices = self.rand_generator.permutation(len(self.Kfoldset[K][phase]))
                            self.Kfoldset[K][phase] = [self.Kfoldset[K][phase][i] for i in indices]
                            self.Kfoldset[K][phase] = [self.Kfoldset[K][phase][i] for i in indices]
                
                # 전처리에 사용할 K번째 fold를 명시
                for phase in self.__phase_list: 
                    dataset_path[phase] = self.Kfoldset[self.test_K][phase]

            # 만약 전처리 정보 저장시, 주어진 argument들을 dictionary로 변환.
            if save_info :
                info = self.__attr2dict()
                if self.split_mode == 'kfold':
                    info['Kfoldset'] = self.Kfoldset
                # 만약 전처리 데이터 세트를 저장하지 않을 시 바로 전처리 저장.
                if not save_batchset : 
                    now = datetime.datetime.now()
                    now = '{:04d}{:02d}{:02d}-{:02d}{:02d}_'.format(now.year, now.month, now.day,
                                                                    now.hour, now.minute)
                    self.filename = './'+now+self.filename
                    write_pickle(self.filename, info)

        # 만약 전처리된 데이터가 있다면 호출 단계에서 'batch_set'이 복구 되었을 것이므로 전처리 미수행.
        try : 
            getattr(self,'batch_set')
        # 'batch_set'이라는 attributes가 없다면 전처리를 새롭게 진행.
        except : 
            # 전처리 병렬 수행
            self._parallelized_load(dataset_path)
            
            # 전처리된 데이터 저장
            if save_info and save_batchset : 
                now = datetime.datetime.now()
                now = '{:04d}{:02d}{:02d}-{:02d}{:02d}_'.format(now.year, now.month, now.day,
                                                                now.hour, now.minute)
                self.filename = '/data/processed_dataset/'+now+self.filename
                info['batch_set'] = self.batch_set
                print('Saving Batchset ... ')
                write_pickle(self.filename, info)
                
    def __attr2dict(self):
        '''
        Description :
            "dataset" object가 가지고 있는 전처리 옵션 attributes들을 dictionart 형태로 반환.
            
        Arguments : 
            No Arguments
            
        Output : 
            dictionary 형태로 변환된 전처리 옵션 변수
        '''
        info = {}
        info['path_list'] = deepcopy(self.path_list)
        info['attr'] = {
            'img_size' : self.img_size,
            'normalize' : self.normalize,
            'polar' : self.polar,
            'do_polar_aug' : self.do_polar_aug,
            'polar_augment_count' : self.polar_augment_count,
            'search_angle' : self.search_angle,
            'contrast' : self.contrast,
            'rotation' : self.rotation,
            'rotation_theta_interval' : self.rotation_theta_interval,
            'fliplr' : self.fliplr,
            'shuffle' : self.shuffle,
            'split_ratio' : self.split_ratio,
            'equalize' : self.equalize,
            'sharpening' : self.sharpening,
            'median' : self.median,
            'zoom_in' : self.zoom_in,
            'select_channel' : self.select_channel,
            'split_mode' : self.split_mode,
            'batch_size' : self.batch_size,
            'random_seed' : self.random_seed,
            'num_total_K' : self.num_total_K,
            'test_K' : self.test_K
        }
        if self.split_mode == 'kfold':
            info['Kfoldset'] = self.Kfoldset
        return info
    
    def repreprocess(self, do_save=True):
        '''
        Description :
            기존의 전처리된 호출하거나 이미 전처리를 마친 데이터세트를 다시 처리함
            
        Arguments : 
            - do_save (기본값 : True) : 재전처리후 저장 여부
            
        '''
        self.batch_set.clear()
        del self.batch_set
        
        # 만약 K가 변했다면 호출할 리스트 변경
        dataset_path = {phase : [] for phase in self.__phase_list}
        for phase in self.__phase_list: 
            dataset_path[phase] = self.Kfoldset[self.test_K][phase]
        
        # 전처리 병렬 수행
        self._parallelized_load(dataset_path)

        # 전처리된 데이터 저장
        if do_save :
            info = self.__attr2dict()

            now = datetime.datetime.now()
            now = '{:04d}{:02d}{:02d}-{:02d}{:02d}_'.format(now.year, now.month, now.day,
                                                            now.hour, now.minute)
            self.filename = './'+now+self.filename
            info['batch_set'] = self.batch_set
            print('Saving Batchset ... ')
            write_pickle(self.filename, info)
            
    def change_K_fold(self, test_K, batch_size=None, filename=None, reload=False, save_batchset=False):
        '''
        Description :
            현재 전처리세트를 K 번째 분할 세트로 변환
            
        Arguments : 
            - test_K : 변환할 K
            - batch_size ( 기본값 : None ) : 변경할 batch_size
            - filename ( 기본값 : None ) : 변경할 파일 이름
            - reload ( 기본값 : False ) : 재전처리 여부
            - save_batchset ( 기본값 : False ) : 전처리된 데이터 저장여부
        '''
        if filename == None and save_batchset :
            raise ValueError('insert the filename that you want!')
            
        # Attributes 변경
        self.filename = filename
        self.test_K = test_K
        if batch_size != None:
            self.batch_size = batch_size
            
        # 재전처리 진행
        if reload :
            self.repreprocess(save_batchset)

    def __process_data(self, img, roi=None):
        '''
        Description :
            실제 전처리가 진행되는 파이프 라인으로, 전처리 순서를 변경하고 싶다는 해당 함수를 수정해서 진행하는 것을 추천.
            
        Arguments : 
            - img : 전처리할 영상
            - roi : 시신경 유두가 위치한 mask 영상 (search_angle에 필수.)
            
        Output : 
            전처리된 데이터 
        '''
        # 극좌표계변환 수행
        if self.polar :
            img_size = None if self.zoom_in else self.img_size
            
            # Polar Transformation을 사용한 augmentation 수행
            if self.do_polar_aug:
                img_list = polar_transformer(img, roi, img_size=img_size, search_angle=self.search_angle,
                                        do_augment=self.do_polar_aug, augment_count=self.polar_augment_count)
            # Polar Transformation만 수행
            else :
                img_list = [polar_transformer(img, roi, img_size=img_size, search_angle=self.search_angle)]
        else :
            img_list = [transform.resize(img, self.img_size)]
        
        result = []
        for img in img_list :
            img = img.squeeze()
            # 영상 채도 강화
            img = enhance_intensity(img) if self.contrast else img
            # 영상 정규화
            img = normalize_img(img) if self.normalize else img
            # 중간값 필터링
            img = median_filter(img, window=(5,5)) if self.median else img
            # Sharpening 필터링
            img = sharpening_filter(img) if self.sharpening else img
            # Histogram Equalization 필터링
            img = equalize_img(img) if self.equalize else img
            # 영상 색채널 추출
            if img.shape[-1] == 3 and self.select_channel != 'rgb' and self.select_channel in self.__rgb_list:
                img = img[:,:,self.__rgb_list.index(self.select_channel)]
            result.append(img)
            
        return result[0] if len(result) == 1 else result
    
    def _pipeline(self, path):
        '''
        Description :
            전처리를 실제로 수행하는 '__process_data'함수를 병렬처리하기 위해 나머지 필요한 기능들을 함수화하여
            path만을 받아 병렬처리 함수가 간단히 수행될 수 있도록하기 위해 작성된 함수. 전처리외의 데이터 Augmentation이
            주로 수행된다.
            
        Arguments : 
            path : 전처리하기 위한 이미지 영상 데이터 경로
            
        Output : 
            전처리와 증강이 이루어진 데이터 
        '''
        mask_path = path.replace('images', 'masks')
        # 
        label = 0 if self.__label_list[0] in path.split('/')[-2] else 1

        img = io.imread(path)
        roi = io.imread(mask_path)[:,:,0]

        # Augmentation - 회전
        if self.rotation : 
            augmented_img = []
            augmented_lbl = []
            
            theta_set = list(range(0,360,self.rotation_theta_interval))

            for theta in theta_set :
                # 원본 영상 회전 후 전처리
                rotated_img = rotate(img, theta, reshape=False) if theta not in [0, 360] else img
                processed = self.__process_data(rotated_img, roi)
                
                # 데이터 확대 여부
                if self.zoom_in:
                    result = transform.resize(processed, self.img_size)
                    augmented_img.append(result[np.newaxis])
                    augmented_lbl.append(label)
                    for ratio in [0.1, 0.15, 0.2]:
                        zoomed = zoomIN(processed, ratio)
                        zoomed = transform.resize(zoomed, self.img_size)
                        augmented_img.append(zoomed[np.newaxis])
                        augmented_lbl.append(label)
                else :
                    cropped = transform.resize(processed, self.img_size)
                    augmented_img.append(processed[np.newaxis])
                    augmented_lbl.append(label)
            
            # 전처리 및 증강된 데이터 세트
            augmented_img = np.concatenate(augmented_img, axis=0)
            augmented_lbl = np.array(augmented_lbl)
        
        else : 
            # 전처리 수행
            processed = self.__process_data(cropped, roi)

            # 데이터 확대 여부
            if self.zoom_in:
                for img in processed:
                    result = transform.resize(img, self.img_size)
                    augmented_img = [result[np.newaxis]]
                    augmented_lbl = [label]
                    for ratio in [0.1, 0.12, 0.14, 0.16, 0.18, 0.2]:
                        zoomed = zoomIN(img, ratio)
                        zoomed = transform.resize(zoomed, self.img_size)
                        augmented_img.append(zoomed[np.newaxis])
                        augmented_lbl.append(label)
                
                # 전처리 및 증강된 데이터 세트 
                augmented_img = np.concatenate(augmented_img, axis=0)
                augmented_lbl = np.array(augmented_lbl)

            else :
                # 전처리된 데이터 세트
                augmented_img = np.array(processed)
                augmented_lbl = np.array([label])
                
        # Augmentation -  영상 미러링
        if self.fliplr : 
            filplr_img = []
            filplr_lbl = []

            for img, ans in zip(augmented_img, augmented_lbl):
                # 정상과 녹내장 간의 데이터 수 불균형이 존재하여 녹내장 데이터만 미러링을 수행.
                # 만약 전처 데이터 세트에 대해서 미러링을진행하고 싶다면 아래의 조건문을 지울 것.
                if ans == 1 :
                    filplr_img.append(img.copy()[:,::-1][np.newaxis])
                    filplr_lbl.append(ans)
                    
            if len(filplr_lbl) != 0 :
                filplr_img = np.concatenate(filplr_img)
                filplr_lbl = np.array(filplr_lbl)
                augmented_img = np.concatenate([augmented_img, filplr_img])
                augmented_lbl = np.concatenate([augmented_lbl, filplr_lbl])
            
        gc.collect()
        return augmented_img, augmented_lbl
    
    def _parallelized_load(self, dataset_path):
        '''
        Description :
            주어진 데이터세트 경로들을 호출해와 병렬로 처리해주는 함수.
            
        Arguments : 
            dataset_path : 전처리할 데이터 세트의 경로 리스트 모음. 'train'과 'valid'에 대한 각각의 리스트가 존재해야함.
        '''
        
        # 병렬처리될 데이터 저장
        train_dataset = {name:[] for name in self.__data_type}
        valid_dataset = {name:[] for name in self.__data_type}
        
        for phase in self.__phase_list:
            # 병렬처리함수와 progress bar
            multiprocessor = Pool(processes=self.workers)
            pbar = tqdm(total=len(dataset_path[phase]))
            pbar.set_description('[ '+phase.upper()+ ' ] dataset Load and Preproecssing ')

            # 각각의 Thread 종로후 얻은 결과를 처리하기 위한 update함수
            def __update(result):
                img_list, label_list = result
                if phase == 'train' :
                    train_dataset['image'].append(img_list)
                    train_dataset['label'].append(label_list)
                else:
                    valid_dataset['image'].append(img_list)
                    valid_dataset['label'].append(label_list)
                pbar.update(1)
            
            # 비동기병렬처리
            for path in dataset_path[phase]: 
                multiprocessor.apply_async(self._pipeline, args=(path,), callback=__update)
            multiprocessor.close()
            multiprocessor.join()
            pbar.close()
        
        # 병렬처리가 완료된 전처리 데이터
        train_dataset['image'] = np.concatenate(train_dataset['image'], axis=0)
        train_dataset['label'] = np.concatenate(train_dataset['label'], axis=0)
        valid_dataset['image'] = np.concatenate(valid_dataset['image'], axis=0)
        valid_dataset['label'] = np.concatenate(valid_dataset['label'], axis=0)
            
        # 이미지와 Label을 phase와 해당하는 label로 나누어 저장
        self.image = { phase : {label : [] for label in self.__label_list} for phase in self.__phase_list}
        self.label = deepcopy(self.image)
        for img, lbl in zip(train_dataset['image'], train_dataset['label']):
            name = self.__label_list[int(lbl)]
            self.image['train'][name].append(img[np.newaxis])
            self.label['train'][name].append(lbl)
        for img, lbl in zip(valid_dataset['image'], valid_dataset['label']):
            name = self.__label_list[lbl]
            self.image['valid'][name].append(img[np.newaxis])
            self.label['valid'][name].append(lbl)
        
        train_dataset.clear()
        valid_dataset.clear()
        del train_dataset, valid_dataset
        gc.collect()
        
        # 주어진 batch_size가 있다면 전체 전처리된 데이터 세트를 batch로 분할 진행
        # batch로 분할 시에 한 batch 안에 두 클래스가 반씩 들어갈 수 있도록 하였음
        self.batch_set = {}
        if self.batch_size != None:
            for phase in self.__phase_list : 
                self.batch_set[phase] = {dtype : [] for dtype in self.__data_type}
                for label in self.__label_list :
                    self.image[phase][label] = np.concatenate(self.image[phase][label])
                    self.label[phase][label] = np.array(self.label[phase][label])
                    
                    data_length = len(self.label[phase][label])
                    rest = int(data_length % (self.batch_size/2))
                    num_batchs = int(data_length//(self.batch_size/2))
                    
                    if rest == 0:
                        self.image[phase][label] = np.array_split(self.image[phase][label], num_batchs)
                        self.label[phase][label] = np.array_split(self.label[phase][label], num_batchs)
                    elif rest != 0:
                        rest_img_batch = self.image[phase][label][-rest:].copy()
                        self.image[phase][label] = np.array_split(self.image[phase][label][:-rest], num_batchs)
                        self.image[phase][label].append(rest_img_batch)
                        
                        rest_lbl_batch = self.label[phase][label][-rest:].copy()
                        self.label[phase][label] = np.array_split(self.label[phase][label][:-rest], num_batchs)
                        self.label[phase][label].append(rest_lbl_batch)
                
                # 데이터 수의 균형을 맞추기 위해 어떤 종류의 데이터가 많은지 확인
                normal_len = len(self.label[phase]['normal'])
                glaucoma_len = len(self.label[phase]['Glaucoma'])

                if normal_len > glaucoma_len:
                    over_cls = 'normal'
                elif glaucoma_len > normal_len:
                    over_cls = 'Glaucoma'

                # 만약 두 클래스간 균형이 맞지 않는다면, 갯수가 더 많은 클래스의 영상을 제거하여 균형을 맞춤
                if glaucoma_len != normal_len:
                    self.image[phase][over_cls].pop()
                    self.label[phase][over_cls].pop()

                    gap = abs(normal_len-glaucoma_len)
                    if gap != 0:
                        reduce_num = self.rand_generator.choice(normal_len, gap)
                        for idx in reversed(sorted(reduce_num)):
                            self.image[phase][over_cls].pop(idx)
                            self.label[phase][over_cls].pop(idx)                    
                
                # 두 클래스간 batch를 병합하여 batch_size를 만족시킨다
                for data in zip(self.image[phase]['normal'], self.image[phase]['Glaucoma']):
                    self.batch_set[phase]['image'].append(np.concatenate(data))
                for data in zip(self.label[phase]['normal'], self.label[phase]['Glaucoma']):
                    self.batch_set[phase]['label'].append(np.concatenate(data))

                # batch 내부의 데이터 셔플
                if self.shuffle:
                    for i in range(len(self.batch_set[phase]['label'])):
                        indices = self.rand_generator.permutation(len(self.batch_set[phase]['label'][i]))
                        self.batch_set[phase]['image'][i] = self.batch_set[phase]['image'][i][indices]
                        self.batch_set[phase]['label'][i] = self.batch_set[phase]['label'][i][indices]

            del self.image, self.label

        gc.collect()
        
    def print_settings(self):
        '''
        Description :
            현재 전처리 및 그 외 사용중인 Attributes들의 내용 전체를 출력.
        ''' 
        print('##### Settings of Preprocessing #####')
        print('[ image size ] : {}'.format(self.img_size))
        print('[ normalize ] : {}'.format(self.normalize))
        print('[ polar transformation ] : {}'.format(self.polar))
        print('[ polar Augmentation ] : {}'.format(self.do_polar_aug))
        print('[ polar Augmentation Count ] : {}'.format(self.polar_augment_count))
        print('[ search angle ] : {}'.format(self.search_angle))
        print('[ contrast ] : {}'.format(self.contrast))
        print('[ rotation ] : {}'.format(self.rotation))
        print('[ rotation theta interval ] : {}'.format(self.rotation_theta_interval))
        print('[ mirroring ] : {}'.format(self.fliplr))
        print('[ shuffle ] : {}'.format(self.shuffle))
        print('[ split ratio ] : {}'.format(self.split_ratio))
        print('[ workers ] : {}'.format(self.workers))
        print('[ equalization ] : {}'.format(self.equalize))
        print('[ sharpening ] : {}'.format(self.sharpening))
        print('[ median filter ] : {}'.format(self.median))
        print('[ Zoom in ] : {}'.format(self.zoom_in))
        print('[ select channel ] : {}'.format(self.select_channel))
        print('[ split mode ] : {}'.format(self.split_mode))
        print('[ batch size ] : {}'.format(self.batch_size))
        print('[ random seed ] : {}'.format(self.random_seed))
        if self.split_mode == 'kfold':
            print('[ K ] : {}'.format(self.num_total_K))
        print('#####################################')
        
    def change_batch_size(self, new_batch_size, filename=None, do_save=False):
        '''
        Description :
            기존의 batch로 분할된 데이터 세트를 새로운 batch size로 변환
            
        Arguments : 
            - new_batch_size : 새롭게 변경하고자 하는 batch_size
            - filenames (기본값 : None) : 저장할 파일 이름
            - do_save (기본값 : False) : 저장 여부
        '''
        
        self.batch_size = new_batch_size
        
        # 기존의 분할된 batch를 전부 통합
        self.batch_set = {phase:{'image':np.concatenate(self.batch_set[phase]['image']),
                                 'label':np.concatenate(self.batch_set[phase]['label'])} 
                          for phase in self.__phase_list}
        
        # 이전에 데이터를 분할하기 전으로 데이터들을 각각의 클래스에 맞게 분할하여 재정렬
        all_images = { phase : {label : [] for label in self.__label_list} for phase in self.__phase_list}
        all_labels = deepcopy(all_images)

        for phase in self.__phase_list:
            for img, lbl in zip(self.batch_set[phase]['image'], self.batch_set[phase]['label']):
                name = self.__label_list[lbl]
                all_images[phase][name].append(img[np.newaxis])
                all_labels[phase][name].append(lbl)
                
            for name in self.__label_list:
                all_images[phase][name] = np.concatenate(all_images[phase][name])
                all_labels[phase][name] = np.array(all_labels[phase][name])

        # Phase 별로 batch 분할 재진행. 한 batch 안에 두 클래스가 반씩 들어갈 수 있도록 하였음
        for phase in self.__phase_list : 
            self.batch_set[phase] = {dtype : [] for dtype in self.__data_type}
            for label in self.__label_list :
                data_length = len(all_labels[phase][label])
                rest = int(data_length % (self.batch_size/2))
                num_batchs = int(data_length//(self.batch_size/2))
                if rest == 0:
                    all_images[phase][label] = np.array_split(all_images[phase][label], num_batchs)
                    all_labels[phase][label] = np.array_split(all_labels[phase][label], num_batchs)
                elif rest != 0:
                    rest_img_batch = all_images[phase][label][-rest:].copy()
                    all_images[phase][label] = np.array_split(all_images[phase][label][:-rest], num_batchs)
                    all_images[phase][label].append(rest_img_batch)

                    rest_lbl_batch = all_labels[phase][label][-rest:].copy()
                    all_labels[phase][label] = np.array_split(all_labels[phase][label][:-rest], num_batchs)
                    all_labels[phase][label].append(rest_lbl_batch)

            # 데이터 수의 균형을 맞추기 위해 어떤 종류의 데이터가 많은지 확인
            normal_len = len(all_labels[phase]['normal'])
            glaucoma_len = len(all_labels[phase]['Glaucoma'])

            if normal_len > glaucoma_len:
                over_cls = 'normal'
            elif glaucoma_len > normal_len:
                over_cls = 'Glaucoma'

            # 만약 두 클래스간 균형이 맞지 않는다면, 갯수가 더 많은 클래스의 영상을 제거하여 균형을 맞춤
            if glaucoma_len != normal_len:
                all_images[phase][over_cls].pop()
                all_labels[phase][over_cls].pop()

                gap = abs(normal_len-glaucoma_len-2)
                if gap != 0:
                    reduce_num = self.rand_generator.choice(normal_len, gap)
                    for idx in reversed(sorted(reduce_num)):
                        all_images[phase][over_cls].pop(idx)
                        all_labels[phase][over_cls].pop(idx)                    

            # 두 클래스간 batch를 병합하여 batch_size를 만족시킨다
            for data in zip(all_images[phase]['normal'], all_images[phase]['Glaucoma']):
                self.batch_set[phase]['image'].append(np.concatenate(data))
            for data in zip(all_labels[phase]['normal'], all_labels[phase]['Glaucoma']):
                self.batch_set[phase]['label'].append(np.concatenate(data))

            # batch 내부의 데이터 셔플
            if self.shuffle:
                for i in range(len(self.batch_set[phase]['label'])):
                    indices = self.rand_generator.permutation(len(self.batch_set[phase]['label'][i]))
                    self.batch_set[phase]['image'][i] = self.batch_set[phase]['image'][i][indices]
                    self.batch_set[phase]['label'][i] = self.batch_set[phase]['label'][i][indices]
        
        # 기존 재정렬을 위해 사용되었던 변수들 삭제
        all_images.clear()
        all_labels.clear()
        del all_images, all_labels
        gc.collect()
        
        # 결과 저장
        if do_save :
            info = self.__attr2dict()
            info['batch_set'] = self.batch_set

            now = datetime.datetime.now()
            now = '{:04d}{:02d}{:02d}-{:02d}{:02d}_'.format(now.year, now.month, now.day,
                                                            now.hour, now.minute)
            try : 
                getattr(self,'filename')
            except : 
                if filename is None:
                    self.filename = 'GD_phase_division.pkl'
                else :
                    self.filename = filename
                self.filename = '/data/processed_dataset/'+now+self.filename
            print('Saving Batchset ... ')
            write_pickle(self.filename, info)
