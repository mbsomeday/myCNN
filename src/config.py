import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--img_dir', type=str, default=r'../../PetImages')
parser.add_argument('--img_size', type=int, default=(180, 180))
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--channels', type=int, default=3)



cfg = parser.parse_args()














