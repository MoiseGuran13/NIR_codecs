# Neuromrophicly Inspired Representation (NIR) for general video codecs
---
## Installation

This repository has been developed with Miniconda 24.11.3 and Python 3.10. We recommend using a conda enviroment to keep this project isolated. Here is a guide for installing [conda](https://www.anaconda.com/docs/getting-started/miniconda/install).

```bash
conda create -n nir python=3.10
conda activate nir
```
After cloning the repository, all but one dependency can be installed via the requirements.txt file. 

```bash
pip install -r requirements.txt
```

Last but not least, an installation of PyTorch 2.5.1 will be required. This is dependent on the gpu of the user, but here is the [official PyTorch guide.](https://pytorch.org/get-started/locally/)


## Composition

This repository makes heavy use of code forked from two others. The [v2e](https://github.com/SensorsINI/v2e) toolbox and the [evreal](https://github.com/ercanburak/EVREAL) codebase.

All purely original code can be found in the encode.py and the decode.py files as well as the notebooks folder. Other files may present minute alterations designed specifically for this project.

All v2e content can be found the in v2e.py file and the v2ecore folder. The remaining file structure resembles that of evreal.

## Usage

### Encoding

In order to encode a video, the encode.py file and function can be used. While support exists for all video formats as of 2024, only the mp4 format has been properly tested. 

```
usage: encode.py [-i input video] 
                [-o output file name]
                [-op output path] 
                [-t threshold]
                [-s sigma treshold]
    
mandatory arguement:
  -i    Relative path to the input video

optional arguements:
  -o    Name of the output file, excluding the ".aedat4" file extension
        If this is left empty, the output name will be the same as the input video
  -op   Relative path to the output file
  -t    Positive and Negative treshold of change used for discretizing the video signal
  -s    Sigma treshold used by the v2e toolbox
```

An example of how to encode a video named Beauty.mp4 is:

```bash
python encode.py -i Beauty.mpy -o Beauty -op output -t 0.15 -s 0.03
```

As a result a new directory may be created, if one did not already exist, now containing the Beauty.aedat4 file

### Decoding

Due to the demonstrative nature of this code, and the heavy dependency on using evreal's built-in multi-model approach, the original video is required for decoding using this repository.

Decoding is done using the decode.py file. 

```
usage: decode.py [-i input aedat4 file]
                 [-or original input video]
                 [-o output video name]
                 [-op output path] 
                 [-m model]
    
mandatory arguement:
  -i    Relative path to the input aedat4 file
  -or   Relative path to the original input video

optional arguements:
  -o    Name of the output video, excluding the ".mp4" file extension
        If this is left empty, the output name will be the same as the input aedat4 file
  -op   Relative path to the output file
  -m    Model name used for video reconstruction. Any Unet model with a coresponding json file in the config\method folder will work.
        By default E2VID will be used.
```

An example of how to decode a Beauty.aedat4 file using decode.py is as follows:

```bash
python decode.py -i Beauty.aedat4 -or Beauty.mp4 -o New_Beauty -op output -m E2VID
```

The outcome of this command will be a new folder named "output" if one did not already exist inside which a new New_Beauty.mp4 will be accessible. It is important that if the new video is to be placed in the same directory as the original mp4 video, that these do not share a name.

### The Example given:

<p float="left">
  <img src="/media/original_Beauty.gif" /> 
  <img src="/media/E2VID_outcome.gif" />
</p>