import os
import sys
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait
import functools

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
    import utils.util as util  # noqa: E402

except ImportError as e:
    print(e)

def scale_im(filename, up_scale, mod_scale, sourcedir, saveHRpath,saveLRpath,saveBicpath):
    # read image
    image = cv2.imread(os.path.join(sourcedir, filename))

    width = int(np.floor(image.shape[1] / mod_scale))
    height = int(np.floor(image.shape[0] / mod_scale))
    # modcrop
    if len(image.shape) == 3:
        image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
    else:
        image_HR = image[0:mod_scale * height, 0:mod_scale * width]
    # LR
    image_LR = imresize_np(image_HR, 1 / up_scale, True)
    # bic
    image_Bic = imresize_np(image_LR, up_scale, True)

    cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
    cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
    cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)
    return filename

def generate_mod_LR_bic(options):
    # set parameters
    up_scale = options.downscale
    mod_scale = options.downscale
    # set data dir
    sourcedir = options.datapath
    savedir = options.out

    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(mod_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR')):
        os.mkdir(os.path.join(savedir, 'HR'))
    if not os.path.isdir(os.path.join(savedir, 'LR')):
        os.mkdir(os.path.join(savedir, 'LR'))
    if not os.path.isdir(os.path.join(savedir, 'Bic')):
        os.mkdir(os.path.join(savedir, 'Bic'))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png')]
    num_files = len(filepaths)
    pbar = util.ProgressBar(num_files)

    #### THREADS
    # pool = Pool(options.workers)
    #
    #
    # def mycallback(arg):
    #     '''get the image data and update pbar'''
    #
    #     pbar.update('Downscaled {}'.format(arg))
    #
    # worker_fn = functools.partial(scale_im, up_scale=up_scale, mod_scale=mod_scale,
    #                               sourcedir=sourcedir, saveHRpath=saveHRpath, saveLRpath=saveLRpath,
    #                               saveBicpath=saveBicpath)
    # for file in filepaths:
    #     pool.apply_async(worker_fn, args=(file,), callback=mycallback)
    #pool = Process(options.workers)


    def mycallback(future):
        '''get the image data and update pbar'''
        fname = future.result()
        pbar.update('Downscaled {}'.format(fname))
    worker_fn = functools.partial(scale_im, up_scale=up_scale, mod_scale=mod_scale,
                                  sourcedir=sourcedir, saveHRpath=saveHRpath, saveLRpath=saveLRpath,
                                  saveBicpath=saveBicpath)
    futures = []
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        for file in filepaths:
            future = pool.submit(worker_fn, file)
            future.add_done_callback(mycallback)
            futures.append(future)
        wait(futures)

    print('Done Downsampling {} images.\n'.format(num_files))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate lowres")
    parser.add_argument('--datapath', type=str, required=True, help='')
    parser.add_argument('--out', type=str, required=True, help='')
    parser.add_argument('--downscale', type=int, required=False, default=2, help='')
    parser.add_argument("--workers", type=int, default=4, required=False)

    generate_mod_LR_bic(parser.parse_args())
