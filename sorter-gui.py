#!/usr/bin/env python3

"""A simple GUI for sorter.py."""

import kivy
kivy.require('1.9.1')

from os                   import path
from ast                  import literal_eval
from PIL                  import Image
from multiprocessing      import Pool
from kivy.app             import App
from kivy.clock           import Clock
from kivy.uix.label       import Label
from kivy.uix.button      import Button
from kivy.uix.checkbox    import CheckBox
from kivy.uix.textinput   import TextInput
from kivy.uix.boxlayout   import BoxLayout
from kivy.uix.gridlayout  import GridLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.filechooser import FileChooserListView
from sorter               import sort


def interp(thing):
    """Creates a Python object from the given object string representation."""
    try:
        return literal_eval(str(thing))
    except ValueError:
        return str(thing)


class MainScreen(BoxLayout):
    """Main view."""

    def updateProgress(self, value):
        """Progress bar updating. Only to be called by clock schedulling."""
        self.progressBar.value = value

    def compute(self):
        """Main computing fuction."""
        images = [Image.open(path.join(self.fileSelector.path, selectedItems)) for selectedItems in self.fileSelector.selection]

        Clock.schedule_once(lambda dt: self.updateProgress(5)) # NOTE: PBar

        kwargs = [{
            'image':                image,
            'ponderation':          interp(self.ponderationInput.text),
            'maxChunkSizeRange':    interp(self.maxChunkSizeRangeInput.text),
            'threshold':            int(self.thresholdInput.text),
            'transpose':            self.transposeInput.active,
            'alternativeThreshold': self.alternativeThresholdInput.active,
            'alternativeReverse':   self.alternativeReverseInput.active,
            'name':                 self.fileSelector.selection[i]
        } for i, image in enumerate(images)]

        sort(**kwargs[0])

        Clock.schedule_once(lambda dt: self.updateProgress(10)) # NOTE: PBar

        # pool = Pool(int(self.threadsInput.text))
        # pool.starmap(sort, kwargs)
        # pool.close()
        # pool.terminate()

        Clock.schedule_once(lambda dt: self.updateProgress(100)) # NOTE: PBar


class SorterGUI(App):
    """Main app."""
    def build(self):
        return MainScreen()


if __name__ == '__main__':
    SorterGUI().run()
