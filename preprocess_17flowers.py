import logging

from pathlib import Path

logging.basicConfig(
    format='%(asctime)s - %(module)s - %(levelname)-6s - %(message)s',
    level=logging.DEBUG)

# constant
origin_path = Path('./dataset/17flowers/origin')
output_path = Path('./dataset/17flowers/train')
flower_images_count_each_class = 80

image_pathes = sorted(origin_path.glob('*.jpg'))
# 17 class flowers, each one has 80 images, total 1360 image
logging.info('split into 17 classes...')
classes = [
    'daffodil', 'snowdrop', 'lilyvalley', 'bluebell', 'crocus', 'iris',
    'tigerlily', 'tulip', 'fritillary', 'sunflower', 'daisy', 'coltsfoot',
    'dandelion', 'cowslip', 'buttercup', 'windflower', 'pansy'
]

for no, image_path in enumerate(image_pathes):
    class_no = no // flower_images_count_each_class
    class_name = classes[class_no]
    logging.debug('copy No. %d image %r into class %r folder.', no,
                  image_path.name, class_name)

    class_path = (output_path / class_name)
    if not class_path.exists():
        class_path.mkdir(parents=True)
    output_path = (class_path / image_path.name)

    # copy image
    data = image_path.read_bytes()
    output_path.write_bytes(data)

logging.info('preprocess done!')
