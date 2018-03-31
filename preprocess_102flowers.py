import logging

from pathlib import Path
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# constant
origin_path = Path('./dataset/102flowers/origin')
labels_path = Path('./dataset/102flowers/imagelabels.mat')
output_path = Path('./dataset/102flowers/train')

matdata = loadmat(labels_path)
int_labels = list(matdata['labels'][0, :])

le = LabelEncoder()
int_labels = le.fit_transform(int_labels)

image_pathes = sorted(origin_path.glob('*.jpg'))
# 107 classes of flowers, each class has different size of dataset.
labels = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells',
    'sweet pea', 'english marigold', 'tiger lily', 'moon orchid',
    'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
    "colt's foot", 'king protea', 'spear thistle', 'yellow iris',
    'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
    'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary',
    'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
    'stemless gentian', 'artichoke', 'sweet william', 'carnation',
    'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
    'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip',
    'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
    'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
    'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower',
    'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
    'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone',
    'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum',
    'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania',
    'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory',
    'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani',
    'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow',
    'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum',
    'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
    'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper',
    'blackberry lily'
]

logging.info('split into %d classes...', len(labels))

for no, image_path in enumerate(image_pathes):
    int_label = int_labels[no]
    label = labels[int_label]
    logging.debug('copy No. %d image %r into class %r folder.', no,
                  image_path.name, label)
    data = image_path.read_bytes()
    label_path = output_path / str(label)
    if not label_path.exists():
        label_path.mkdir(parents=True)
    (label_path / image_path.name).write_bytes(data)

logging.info('preprocess done!')
