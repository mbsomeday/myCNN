import os
from config import cfg
import numpy as np
from tensorflow import keras
import abc
import cv2


class Generator(keras.utils.Sequence):
    
    def __init__(self, img_dir, batch_size, img_size, channels, shuffle=True):
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.nb_samples, self.paths_cls = self._get_nbSamples()
        self.img_size = img_size
        self.shuffle = shuffle
        self.channels = channels
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(self.nb_samples / self.batch_size))
    
    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_pathsAndCls = []
        for i in indices:
            batch_pathsAndCls.append(self.paths_cls[i])
        
        batch_imgs, batch_labels = self.get_batch_data(batch_pathsAndCls)
        
        return batch_imgs, batch_labels
    
    def get_batch_data(self, batch_batch_pathsAndCls):
        x = np.zeros(shape=(self.batch_size, *self.img_size, self.channels), dtype=np.float)
        y = np.zeros(shape=(self.batch_size,), dtype=int)
        
        for idx, item in enumerate(batch_batch_pathsAndCls):
            y[idx] = item[0]
            img = self.preprocess_img(item[1])
            x[idx] = img
        return x, y
            
    
    def preprocess_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=self.img_size)
        img = img / 255.0
        img = img[np.newaxis, :]
        return img
        
        
    
    def _get_nbSamples(self):
        l = 0
        path_list = []
        for subClassDir in os.listdir(self.img_dir):
            cls = 0 if subClassDir == "Cat" else 1
            sub_path = os.path.join(self.img_dir, subClassDir)
            l += len(os.listdir(sub_path))
            for img in os.listdir(sub_path):
                img_path = os.path.join(sub_path, img)
                path_list.append((cls, img_path))
        return l, path_list
    
    def on_epoch_end(self):
        self.indices = np.arange(self.nb_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def test(self):
        return self.paths_cls


if __name__ == '__main__':
    g = Generator(cfg.o_img_dir, cfg.batch_size, cfg.img_size, cfg.channels, shuffle=False)
    # for i in range(len(g)):
    #     print(i)
    #     images, _ = g[i]
    #
    