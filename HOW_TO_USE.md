# Tasks

1. Project Set Up
    - **Option 1:** Project Installation/Initialization (On first use)
    - **Option 2:** Subsequent Environment Setup (On subsequent use)
1. Test on a single image
1. Detection of batch images and insertion into database
1. Main application for use

## Breakdown

**If you're running the project for the first time, perform Project Installation/Initialization, subsequently perform Project Setup**

### Project Installation/Initialization: SHOULD BE DONE ONLY ONCE ON PROJECT CREATION (Option 1)
> **NOTICE:** If you have already successfully done this whole initialization task before, use Option 2. Reinitializing will corrupt the whole codespace and you have to start afresh again.

**Step 1:** Open terminal and run the following commands, **sequentially**
- `conda create --name fashionv python=3.8 -y` *- fashionv is the name of our virtual environment*
- `conda init`

**Step 2:** Close terminal and open again, and run these commands
- `conda activate fashionv`
- `conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch`
- `pip install -U openmim`
- `mim install mmcv-full==1.4.0`
- `pip install mmdet==2.18.0`
- `pip install git+https://github.com/cocodataset/panopticapi.git`
- `pip3 install -U scikit-learn scipy matplotlib scikit-image chainercv`
- `pip install opencv-python-headless`
> Take a deep breath, if the above commands ran successfully, you're almost there. If not, you have to seek solution to ensure it works before proceeding.

**Step 3:** Now run these commands
- `mkdir data && cd data`
- `pip install gdown`
- `gdown --id 1dHklkhGgxWjWEeHoUkQqyFWZRtbD2Vyu`
> **IMPORTANT:** Right click and rename the file that has been downloaded inside the main project folder (FASHIONMASK), change the name from `fashionformer_r101_3x.pth` to `model_final.pth`

**Step 4:** Now run the next set of commands
- `mv model_final.pth data/`
- `mkdir Fashionpedia && cd Fashionpedia`
- `mkdir train && mkdir test && mkdir annotations`
- `cd ../../mmdetection`

And finally, run
- `pip install webcolors streamlit streamlit_extras`

> Now we're set! 
> **NOTICE:** Once you have successfully done this initialization task, you will not do it AGAIN unless you are creating another new codespace. Initialization is complete for this workspace, the next time, just use Option 2.

### Getting Ready for Detection: SHOULD BE DONE SUBSEQUENTLY FOR PROJECT SETUP (Option 2)
- Open terminal
- Activate virtual environment, if  using `conda activate fashionv`
- Navigate to 'mmdetection' folder using `cd mmdetection`

### Task 1: Test on a single image
- Run the code below in terminal, where demo/girl.png is the image to be detected
- `PYTHONPATH='.' python demo/ntt.py ../all_test/girl.png configs/fashionformer/fashionpedia/fashionformer_r101_mlvl_feat_8x.py ../data/model_final.pth --device cpu --out-file ../outputx/demoe.png --score-thr 0.5`
- demo/ntt.py is the script to perform the detection
- demo/girl.png is the file to be detected
- configs/fashionformer/fashionpedia/fashionformer_r101_mlvl_feat_8x.py is the config file
- ../data/model_final.pth is the model weight
The rest settings should remain
- Output image of detection will be found inside the mmdetection/outputx in demoe.png file

### Task 2: Add to database
- Run the code below in terminal, where batch is the folder that contains all the images to be added
- `PYTHONPATH='.' python demo/process_db.py ../all_batch configs/fashionformer/fashionpedia/fashionformer_r101_mlvl_feat_8x.py ../data/model_final.pth --device cpu --score-thr 0.5`
- The all_batch folder contains all the images to be detected and inserted to database
- The output images can be found in `outputx` folder, inside the `mmdetection` folder

### Task 3: Running the Main App
- Open terminal and run the following code:
- `PYTHONPATH='.'  streamlit run demo/app.py`
- Accept option to make app public
- Switch to ports tab in your terminals. Hover on the 'Forwarded Address' and press the web icon to open on a browser, or copy it by pressing the copy icon.



