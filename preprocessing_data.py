import numpy as np
import imageio.v3 as iio
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
train_ratio = 70 # We know there are 100 files per class

def crop_images():
    for class_idx,folder in enumerate(sorted(Path(r"GTZAN_Data\images_original").iterdir())):
        print(class_idx, folder) # Shows arbitrary classs number for the folder the dataset came from
        for img_idx,file in enumerate(Path(f"{folder}").iterdir()):
            if not file.is_file():
                print('Not a file found, skipping')
                continue
            picture=Image.open(file)
            picture_cropped=picture.crop((54, 35, 390, 253))
            picture_cropped.save(f"GTZAN_Data\images_original_cropped\{classes[class_idx]}\\"+classes[class_idx]+str(img_idx).zfill(5)+'.png', "PNG")


def resize_images():
    for class_idx,folder in enumerate(sorted(Path(r"GTZAN_Data\images_original_cropped").iterdir())):
        print(class_idx, folder) # Shows arbitrary classs number for the folder the dataset came from
        for img_idx,file in enumerate(Path(f"{folder}").iterdir()):
            if not file.is_file():
                print('Not a file found, skipping')
                continue
            picture=Image.open(file)
            picture=picture.convert('RGB')
            out = picture.resize( [int(0.25 * s) for s in picture.size])
            picture=picture.resize(out.size)
            picture.save(f"GTZAN_Data\images_crop_resized_25\{classes[class_idx]}\\"+classes[class_idx]+str(img_idx).zfill(5)+'.jpg', "JPEG")


def train_test(input_path):
    images = []
    data_train = []
    data_test = []
    counter = 0 
    for class_idx,folder in enumerate(Path(input_path).iterdir()):
        print(class_idx, folder) # Shows arbitrary classs number for the folder the dataset came from
        for img_idx,file in enumerate(Path(f"{folder}").iterdir()):
            if not file.is_file():
                print('Not a file found, skipping')
                continue
            image = iio.imread(file, mode='L')
            images.append(image)
            # Assigning class as first column to the flattened image list
            # Splitting the data into train and test
            if img_idx < train_ratio:
                data_train.append(np.array([class_idx]+list(image.flatten())))
            else:
                data_test.append(np.array([class_idx]+list(image.flatten())))
            counter+=1
        print('Middle Counter:', counter)
    print('End Counter:', counter)
    
    images = np.asarray(images,dtype='int16')
    data_test = np.asarray(data_test, dtype='int16')
    data_train = np.array(data_train, dtype='int16')
    return images, data_train, data_test

if __name__ == "__main__":
    # crop_images()
    # resize_images()

    images, data_train, data_test = train_test(r"GTZAN_Data\images_original")
    print('All original images shape:', images.shape)
    print('All original data train shape:', data_train.shape)
    print('All original data test shape:', data_test.shape)
    print('Single image shape:', images[0].flatten().nbytes)
    # Making sure each class has a 100 records
    print(np.unique(np.array(data_train)[:, 0], return_counts=True)) 
    print(np.unique(np.array(data_test)[:, 0], return_counts=True))

    images, data_train, data_test = train_test(r"GTZAN_Data\images_crop_resized")
    print('All 0.5 resized images shape:', images.shape)
    print('All 0.5 resized data train shape:', data_train.shape)
    print('All 0.5 resized data test shape:', data_test.shape)
    print('Single 0.5 resized image shape:', images[0].flatten().nbytes)
    # Making sure each class has a 100 records
    print(np.unique(np.array(data_train)[:, 0], return_counts=True)) 
    print(np.unique(np.array(data_test)[:, 0], return_counts=True))

    images, data_train, data_test = train_test(r"GTZAN_Data\images_crop_resized_25")
    print('All 0.25 resized images shape:', images.shape)
    print('All 0.25 resized data train shape:', data_train.shape)
    print('All 0.25 resized data test shape:', data_test.shape)
    print('Single 0.25 resized image shape:', images[0].flatten().nbytes)
    # Making sure each class has a 100 records
    print(np.unique(np.array(data_train)[:, 0], return_counts=True)) 
    print(np.unique(np.array(data_test)[:, 0], return_counts=True)) 

    # plt.imshow(images[0])
    # plt.imshow(np.array(data_train[0][1:]).reshape(288, 432)) #Seeing if data reshaped can show same image
    # plt.show()