from PIL import Image
import pickle

for x in range(1,5):
    img_dir='mario-1-{}.png'.format(x)
    im = Image.open(img_dir)
    image = {
        'pixels': im.tobytes(),
        'size': im.size,
        'mode': im.mode,
    }
    file = open('mdata1{}.pkl'.format(x), 'wb', 0)
    pickle.dump(image, file)
    file.close()

# pickle_in = open('mdata.pkl', 'rb')
# example_img = pickle.load(pickle_in)
# print(example_img)
