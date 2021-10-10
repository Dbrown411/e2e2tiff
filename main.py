from oct_converter.readers import E2E
import glob, os
import imageio
import numpy as np
import argparse
from tqdm import tqdm

def e2e_to_tiff(file_name,gamma=1,subcall=False):
    ext = '.tiff'
    folder,file = os.path.split(file_name)
    export_name = f'{folder}{os.sep}{file[:-4]}{ext}'

    with tqdm(total=100,leave=not subcall) as pbar:
        e2e = E2E(file_name)
        pbar.update(25)
        pbar.set_description(f'Reading: {file}')
        oct_volumes = e2e.read_oct_volume()
        pbar.set_description(f'Converting to {ext}...')
        pbar.update(25)
        update_amount = 25/len(oct_volumes)
        for i,oct in enumerate(oct_volumes):
            suffix = f'-{i}' if i>0 else ''
            nframes = len(oct.volume)
            vol_out = np.dstack(oct.volume)
            vol_out = vol_out/256   #renormalize to 1
            vol_out = pow(vol_out,gamma)
            vol_out = vol_out*65535 #scale to 16bit
            vol_out = vol_out.astype(np.uint16)
            fstack_out = [vol_out[:,:,i] for i in range(nframes)]
            imageio.mimwrite(f'{export_name}{suffix}', fstack_out)
            pbar.update(update_amount)
        pbar.update(25)
        tqdm.write(f"{export_name}: Exported")

def convert_folder(folder,gamma=1):
    for f in tqdm(glob.glob(rf'{folder}\*.e2e')):
        e2e_to_tiff(f,gamma=gamma,subcall=True)

def run_tests():
    convert_folder(r".\tests")

if __name__=="__main__":
    curdir = f"{os.path.split(os.path.realpath(__file__))[0]}"
    configs = {
        'initial_dir':curdir
    }
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument('gamma', nargs='?',
                        help='gamma applied to the image',
                        default=0.5, 
                        type=int)

    parser.add_argument('--file', 
                        help="convert a single scan",
                        default=False,
                        action='store_true')
    parser.add_argument('--test', 
                        help="run tests",
                        default=False,
                        action='store_true')                    
    args = parser.parse_args()
    print(args)      
    if args.test:
        run_tests()
    else:
        from tkinter import filedialog
        from tkinter import *
        root = Tk()
        root.withdraw()
        if args.file:
            file_name = filedialog.askopenfilename(initialdir = configs['initial_dir'],
                                        title = 'Select an .e2e file',
                                        filetypes = [("OCT File","*.e2e")]
                                        )
            e2e_to_tiff(file_name)
        else:
            folder_selected = filedialog.askdirectory(parent=root,
                                    initialdir=configs['initial_dir'],
                                    title='Select directory with .e2e Files')
            if folder_selected != '':
                directory = folder_selected
            
            convert_folder(folder_selected)