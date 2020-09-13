# -*- coding: utf-8 -*-

import numpy as np
import polarTransform
import skimage

from scipy.ndimage import *
from skimage import morphology
from scipy.ndimage import binary_opening

def zoomIN(img, zoom_ratio=0.1):
    """
    Description :
        주어진 이미지를 일정 비율 {0.1 ~ 0.5} 범위로 이미지를 확대 할 수 있음.
        
    Arguments :
        - img : 확대할 이미지
        - zoom_ratio (기본값 : 0.1) : 확대할 비율 {0.1 ~ 0.5}
        
    Output :
        확대된 이미지 출력. 이미지 사이즈는 주어진 비율에 따라 다르다.
    """
    
    # 확대 비율 제한 {0.1 ~ 0.5}
    if zoom_ratio < 0 or zoom_ratio > 0.51 :
        raise VauleError('Zoom Ratio must be in 0%')
        
    # 주어진 비율에 따른 width 길이 단위와 hight 길이 단위 연산
    w, h = img.shape[:2]
    w_unit = round(w*zoom_ratio)
    h_unit = round(h*zoom_ratio)
    
    # 우측좌측, 상단하단으로 주어진 width, height 길이 단위만큼 길이 자르기 ( 확대할 영상 크기 연산)
    w -= w_unit*2 
    h -= h_unit*2
    # 확대할 이미지 x, y 시작 지점
    x = w_unit
    y = h_unit
    
    # 이미지 Crop하여 추출
    return img[x:x+w, y:y+h]

def normalize_img(img):
    '''
    Description :
        이미지 영상의 픽셀 값들을 Z-scoring (평균 0, 분산 1) 연산하여 출력 
        
    Arguments :
        - img : 변환할 이미지 입력
    
    Output : 
        Z-score로 변환된 이미지 
    '''
    shape = img.shape # 이미지 크기 가져오기
    img = np.float64(img.reshape(-1)) # 영상을 1d-array로 변환한 후 float 형태로 변환
    img -= img.mean() # 평균 0
    img /= img.std() # 분산 1
    return img.reshape(shape) # 변환된 이미지 출력

def enhance_intensity(img):
    '''
    Description :
        이미지의 색 밝기를 강화, 이미지 색 밝기 분포를 0~255로 변환.
        
    Arguments :
        - img : 변환할 이미지 입력
    
    Output : 
        변환된 이미지
    '''
    return skimage.exposure.rescale_intensity(img) # 색채대비 강화

def equalize_img(img, select_channel='rgb'):
    '''
    Description :
        주어진 영상을 이미지 히스토그램 균일화 (Image Histogram Equalization) 실행하여 이미지의 contrast 향상
        
    Arguments :
        - img : 변환할 이미지 입력
        - select_channel : 입력하는 이미지 색 채널 종류 (기본값 : 'rgb')
    
    Output : 
        균일화가 적용된 영상
    '''
    select_channel = select_channel.lower()
    rgb_list = ['r','g','b'] # 영상 색 채널 : R G B 

    selem = morphology.disk(30)

    # 색 채널을 전부 사용하는 경우, 각 채널마다 Equlization 수행 후 결과 합침
    if select_channel == 'rgb':
        img_eq = []
        for chn in img.transpose(2,0,1):
            img_eq.append(skimage.filters.rank.equalize(chn, selem=selem)[:,:,np.newaxis])
        img_eq = np.concatenate(img_eq, axis=-1)
        
    # 색 채널을 하나만 사용하는 경우,
    elif select_channel in rgb_list:
        chn = rgb_list.index(select_channel)
        img_eq = skimage.filters.rank.equalize(img[:,:,chn], selem=selem)
        
    # 그 외에는 에러처리.
    else :
        raise ValueError('select_channel must be total channels:("rgb") or single channel ("r","g","b").')

    return img_eq

def polar_transformer(img, mask=None, angle=0, img_size=(256,256), margin=False, search_angle=False, interval=5, 
                      do_augment=False, augment_count=1):
    '''
    Description :
        주어진 영상을 데카르트 좌표계에서 극 좌표계로 변환.
        
    Arguments :
        - img : 변환할 이미지
        - mask : Optic Disc Mask 입력. (기본값 : None, 선택옵션, search_angle에 사용함)
        - angle : 데카르트 좌표계 기준으로 극좌표계 변환을 시작할 각도.(기본값 : 0)
        - image_size : 출력할 이미지의 사이즈 (기본값 : (256,256))
        - margin : 극좌표변환후 발생하는 검은 영역 제거 유무. False -> 제거 (기본값 : False)
        - search_angle : 극좌표계 변환을 시작할 각도 탐색 (기본값 : False)
        - interval : 극좌표계 변환을 시작할 각도의 간격 (기본값 : 5)
        - do_augment : 극좌표계 변환하면서 최적 각도를 중심으로 추가 각도로 영상 분할(기본값 : False, Search_angle 활성화 필수)
        - augment_count : Augmentation으로 취할 영상의 갯수, 최적 각도로 양 옆 k개씩 추출. (기본값 : 1)
    
    Output : 
        극 좌표계로 변환된 단일 영상.
        만약 'do_augment' 수행 시, 극좌표계로 변환된 영상 리스트
    '''
    
    if augment_count > 5 :
        raise ValueError('Polar Augmentation can be executed over 5 times bigger.')
    
    # Search_angle은 Mask 이미지를 주어 변환시에 Optic Disc가 잘리지 않는
    # 각도를 찾고, 최대한 Optic Disc가 중앙에 올 수 있는 최적의 각도를 탐색함.
    
    # seach_angle (최적각도탐색) 수행시, 
    angle_range = np.arange(360/interval)*np.pi/(180/interval)
    if search_angle :
        
        # Optic Disc Mask 데이터 존재 유무 체크, 없으면 에러 출력
        if type(mask) == type(None) : 
            raise ValueError('angle searching need optic disc binary mask.')
        
        nearest_angle = None # 최적 각도
        nearest_dist = None # 중앙에서부터의 거리

        # 연산 속도 증가를 위해 기존의 마스크 영상을 1/10 크기만큼으로 줄여
        # binary Mask 이미지를 극좌표계 변환에 사용.
        new_shape = tuple(s//10 for s in mask.shape)
        new_mask = skimage.transform.resize(mask, new_shape)
        new_mask = np.array(binary_opening(new_mask), np.int)
        
        # 최적 각도 탐색 시작 : 0도에서 359도까지 주어진 `interval`에 따라 각도 사용
        for idx, angle in enumerate(angle_range):
            # Mask 이미지 변환
            roi, _ = polarTransform.convertToPolarImage(new_mask, hasColor=False,
                                                        initialAngle=angle, finalAngle=np.pi*2+angle,
                                                        useMultiThreading=True)
            
            # Mask 영상을 projection 시켜 현재 어디에 마스크가 존재하는지
            # 확인할 수 있도록 argmax로 각 열마다 최대가 되는 값(1)이 존재하는
            # 위치 (index)를 반환
            check_roi = roi.argmax(-1) # width축 기준으로 projection
            width_len = roi.shape[0] # width축 길이
            threshold = width_len/20 # 중앙에서부터의 허용범위 설정

            # 마스크 영상 projection 시에 좌측 혹은 우측에 걸쳐 있지 않으면,
            if check_roi[0] == 0 and check_roi[-1] == 0 :
                roi_center = check_roi.argmax() # Optic Disc의 height가 제일 긴 위치를 출력
                width_middle = width_len/2 # width의 중앙
                
                # Optic Disc의 중앙이 이미지 width의 중앙에서부터 허용범위 내에 존재하는 경우,
                if roi_center < width_middle + threshold and roi_center > width_middle - threshold :
                    # 중앙과 Optic Disc 중앙의 거리를 연산
                    dist = abs(roi_center-width_middle)
                    
                    # 만약 이전에 입력된 값이 없는 경우,
                    if nearest_angle == None :
                        nearest_angle = angle
                        nearest_dist = dist
                        nearest_idx = idx
                    # 현재의 거리가 더 짧은 경우,
                    elif nearest_angle != None and nearest_dist > dist:
                        nearest_angle = angle
                        nearest_dist = dist
                        nearest_idx = idx
        
    # Augmentation 진행 시 최적 각도를 중심으로 주어진 갯수 만큼 사용할 각도 추출
    if do_augment :
        min_angle = nearest_idx-augment_count
        max_angle = nearest_idx+augment_count+1
        angle_list = []
        
        if min_angle < 0 :
            angle_list += list(angle_range)[min_angle:]
            angle_list += list(angle_range)[:max_angle]
        elif max_angle > len(angle_range) :
            max_angle -= len(angle_range)
            angle_list += list(angle_range)[min_angle:]
            angle_list += list(angle_range)[:max_angle]
        else :
            angle_list = list(angle_range)[min_angle:max_angle]
        
    # Augmentation이 아닐 경우, 제일 가까운 위치의 각도를 사용
    else : 
        angle_list = [nearest_angle]
          
    img_list = []
    for angle in angle_list:
        # 실제 안저 영상을 Optic Disc가 중앙에 가도록 배치.
        polarImage, _ = polarTransform.convertToPolarImage(img, hasColor=True,
                                                           initialAngle=angle, finalAngle=np.pi*2+angle)
        # width, height 위치 전환
        polarImage = polarImage.transpose(1,0,2)

        # 만약 margin 제거한다면,
        if not margin:
            threshold = 0
            start = int(round(polarImage.shape[0]*0.7))
            for line in polarImage.transpose(1,0,2):
                for idx, pixels in enumerate(line[start:]):
                    if sum(pixels)/len(pixels) < 10:
                        threshold+=idx
                        break

            threshold /= polarImage.shape[1]
            threshold = int(round(threshold))
            polarImage = polarImage[:threshold+start]

        # 만약 이미지 사이즈로 None을 받으면, Resize 수행하지 않음. 
        if img_size is not None : 
            polarImage = skimage.transform.resize(polarImage, img_size)

        img_list.append(polarImage[np.newaxis])
    
    # augmentation 하면서 이미지 사이즈가 다른 경우, 리스트로 반환
    if img_size is None and do_augment:
        return img_list
    # augmentation 하면서 이미지의 사이즈가 동일하게 조절한 경우 하나의 array로 변환하여 반환
    elif img_size is not None and do_augment:
        return np.concatenate(img_list)
    # Augmentation을 수행하지 않았을 경우, 단일 이미지 반환
    else :
        return polarImage

def sharpening_filter(img, alpha=30):
    '''
    Description :
        주어진 이미지의 Edge를 강화. 노이즈없는 상황에서 이미지를 선명하게하는 방법을 보여줌.
        흐림에 필터를 반대로 적용하여 연산
        
    Arguments :
        - img : 변환할 이미지
        - alpha : 흐림에 필터를 반대로 적용할 때 그 정도를 정함. (기본값 : 30)
    
    Output : 
        변환된 이미지
    '''
    blurred_img = gaussian_filter(img, 3)
    filter_blurred_img = gaussian_filter(blurred_img, 1)
    sharpened = blurred_img + alpha * (blurred_img - filter_blurred_img)
    return sharpened

def median_filter(img, window=(30,30), origin=(0,-1)):
    '''
    Description :
        kernel window와 pixel의 값들을 정렬한 후에 중간값을 선택하여 변환.
        salt-and-pepper noise 제거에 가장 효과적.
        
    Arguments :
        - img : 변환할 이미지
        - window : 중간 값을 연산하기 위해 고려할 범위크기 설정 (기본값 : (30,30))
        - origin : 중간 값 연산 시에 적용되는 방향 설정 가능 (기본값 : (0,-1) : 중앙에서 왼쪽으로)
    
    Output : 
        변환된 이미지
    '''
    # 만약 색 채널 전부 사용시,
    if img.shape[-1] == 3 :
        filtered = []
        # 색 채널별로 중간값 필터링 적용 후 다시 RGB 이미지로 합친 뒤 출력
        for chn in img.transpose(2,0,1):
            chn = filters.median_filter(chn, size=window, origin=origin)
            filtered.append(chn[np.newaxis])
        filtered = np.concatenate(filtered, axis=0).transpose(1,2,0)
        return filtered
    
    else :
        return filters.median_filter(img, size=window, origin=origin)