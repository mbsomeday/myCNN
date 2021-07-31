from tensorflow.keras import layers
from config import cfg
from tensorflow import keras
import numpy as np
from data_generator import Generator


def get_model():
    inpt = keras.Input(shape=(*cfg.img_size, cfg.channels))
    x = layers.Conv2D(32, 3, strides=2, padding='same')(inpt)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(64, 4, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    previous_block_activation = x
    
    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        
        x = layers.add(([x, residual]))
        previous_block_activation = x
    
    x = layers.SeparableConv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    activation = 'sigmoid'
    units = 1
    
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inpt, outputs)



if __name__ == '__main__':
    model_path = r'./model/res_model_2.h5'
    model = get_model()
    model(np.ones((1, *cfg.img_size, cfg.channels), dtype="float32"))
    model.load_weights(model_path)
    
    g = Generator(r'../dataset_RNN/PetImages', 100, cfg.img_size, cfg.channels, shuffle=True)
    images, labels = g[0]
    images = images.astype("float32")
    # print("labels:", labels)
    res = model.predict(images)
    r = []
    for i in res:
        cur = 0 if i <= 0.5 else 1
        r.append(cur)
    # print("预测结果：", r)
    total_num = len(res)
    correct_num = 0
    for i in range(total_num):
        if labels[i] == r[i]:
            correct_num += 1
            
    print("总预测数量：", total_num)
    print("预测正确数量：", correct_num)
    





















