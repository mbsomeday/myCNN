import os
import tensorflow as tf
from config import cfg
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from data_generator import Generator


def check_img_type():
    base_dir = cfg.o_img_dir
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(base_dir, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jsif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jsif:
                num_skipped += 1
                os.remove(fpath)
    print("Deleted %d images" % num_skipped)


def generate_subset(cfg):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        cfg.img_dir,
        labels="inferred",
        # class_names=['Dog','Cat'],
        validation_split=0.2,
        subset='training',
        seed=1337,
        image_size=cfg.img_size,
        batch_size=cfg.batch_size
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        cfg.img_dir,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=cfg.img_size,
        batch_size=cfg.batch_size
    )
    
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    # 	for i in range(9):
    # 		ax = plt.subplot(3, 3, i + 1)
    # 		plt.imshow(images[i].numpy().astype("uint8"))
    # 		plt.title(int(labels[i]))
    # 		plt.axis("off")
    # plt.show()
    return train_ds, val_ds


def data_augmentation(train_ds, val_ds):
    data_aug = keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal"),
                                 layers.experimental.preprocessing.RandomRotation(0.1)])
    augmented_train_ds = train_ds.map(lambda x, y: (data_aug(x, training=True), y))
    
    augmented_train_ds = augmented_train_ds.prefetch(buffer_size=cfg.batch_size)
    val_ds = val_ds.prefetch(buffer_size=cfg.batch_size)
    
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    # 	for i in range(9):
    # 		auged_images = data_aug(images)
    # 		ax = plt.subplot(3, 3, i+1)
    # 		plt.imshow(auged_images[0].numpy().astype("uint8"))
    # 		plt.axis("off")
    # plt.show()
    
    return augmented_train_ds, val_ds


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


# 由于tf1.14没有image_dataset_from_directory，只能用这个办法生成数据
def data_generator(cfg):
    pass


if __name__ == '__main__':
#     train_ds, val_ds = generate_subset(cfg)
#     augmented_train_ds, val_ds = data_augmentation(train_ds, val_ds)
    
    model = get_model()
    # model.summary()
    callbacks = [keras.callbacks.ModelCheckpoint(r"../model/res_model.h5")]
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    g = Generator(cfg.o_img_dir, cfg.batch_size, cfg.img_size, cfg.channels)
    model.fit(g, epochs=5, callbacks=callbacks)
    