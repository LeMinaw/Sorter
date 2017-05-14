# Sorter.py
A Python module generating cool pictures by sorting an input image's pixels.
## Demo pictures
Some images the module generated can be found in `/demo/` in this repository.
## How to install
### Python environnement
You need python 3.x correctly installed and configured in order to run this script.
I **strongly** recommend using the 64 bits version if your system architecture have x64 capabilities.
Having `pip` installed is also recommanded, becauses it makes packages installation painless. `pip` should be delivered with the python 3.x installer.
### Dependancies
To install python package `foobar` with pip, you just have to type `pip install foobar` in a terminal with root privileges.
* The PIL, Python Imaging Library. `pip install pillow`
* The multiprocess module. `pip install multiprocess`
* The Kivy GUI library *optional*. `pip install kivy`
## How to use
### GUI
A Graphical User Interface is in progress. However, it's quite buggy and incomplete so I don't recommend using this for the moment.
If your quite crazy, just run `sorter-gui.py` and check it out.
### CLI
In fact, the Command Line Interface is not avalaible yet.
### Python call
For the moment, I recommend importing the main function in a python shell (`from sorter import sort`), or editing `sorter.py` directly.
#### In a python shell
The only function you'll have to call is `sort`. In fact, the simplest call you may do is simple as `sort(Image.open('niceImageFile.jpg'))` (obviously you'll need to do a `from PIL import Image` first).
`sort` can take a lot of optional arguments, each one is described in code documentation.
#### By editing the module
At the end of it, you'll find a `if __name__ ==  '__main__':` statement.
Just edit it to match your needs.
Some exemples are present. You can do various things such as animation, multiprocessing of several images, interactive mode, or video frames computation.
**Warning !** For the moment, all commented exemples are old legacy exemples for reference. Only the uncommented code is working *as is*.
