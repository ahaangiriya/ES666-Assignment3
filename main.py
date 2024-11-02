import pdb
import src
import glob
from importlib import util
import os
import cv2

# Set the path to the images directory
path = 'Images{}*'.format(os.sep)  # Use os.sep for cross-platform compatibility

all_submissions = glob.glob('./src/*')
os.makedirs('./results/', exist_ok=True)

for idx, algo in enumerate(all_submissions):
    print('****************\tRunning Awesome Stitcher developed by: {}  | {} of {}\t********************'.format(
        algo.split(os.sep)[-1], idx + 1, len(all_submissions)))
    try:
        module_name = '{}_{}'.format(algo.split(os.sep)[-1], 'stitcher')
        filepath = '{}{}stitcher.py'.format(algo, os.sep)  # Fixed path concatenation
        spec = util.spec_from_file_location(module_name, filepath)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PanaromaStitcher = getattr(module, 'PanaromaStitcher')
        inst = PanaromaStitcher()

        # Process each set of images
        for impaths in glob.glob(path):
            print('\t\t Processing... {}'.format(impaths))
            stitched_image, homography_matrix_list = inst.make_panaroma_for_images_in(impaths)  # Positional argument

            outfile = './results/{}/{}.png'.format(impaths.split(os.sep)[-1], spec.name)
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            cv2.imwrite(outfile, stitched_image)
            print(homography_matrix_list)
            print('Panorama saved ... @ {}'.format(outfile))
            print('\n\n')

    except Exception as e:
        print('Oh No! My implementation encountered this issue\n\t{}'.format(e))
        print('\n\n')
