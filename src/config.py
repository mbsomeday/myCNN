import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--h_img_dir', type=str, default=r'../../PetImages')
parser.add_argument('--o_img_dir', type=str, default=r'../../dataset_RNN/PetImages')

parser.add_argument('--img_size', type=int, default=(180, 180))
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--channels', type=int, default=3)



cfg = parser.parse_args()














