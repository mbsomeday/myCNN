import os
import tensorflow as tf


base_dir = r'../../PetImages'
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


































