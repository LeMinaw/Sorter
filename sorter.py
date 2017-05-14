#!/usr/bin/env python3

"""Sorter.py \n
Quite hackable pixel sorter."""

# import cProfile

from PIL             import Image, ImageDraw, ImageColor
from math            import exp
from multiprocess    import Pool
from itertools       import chain
from time            import clock
from random          import random, randint
from os              import close, listdir


NORMALIZED_HUE = lambda px: hue(normalize(px))
K = (0, 0, 0)
R = (1, 0, 0)
G = (0, 1, 0)
B = (0, 0, 1)
C = (0, 1, 1)
M = (1, 0, 1)
Y = (1, 1, 0)
W = (1, 1, 1)


def randstr(size=16):
    """Returns a random string of size chars."""
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    string = ""
    for i in range(16):
        string += chars[randint(0, len(chars)-1)]
    return string


def normalize(pixel):
    """Return a normalised tuple from pixel data."""
    return tuple(color / 255.0 for color in pixel)


def luminance(pixel):
    """Returns the luminance of a normalised pixel."""
    return (max(pixel) + min(pixel)) / 2.0


def saturation(pixel):
    """Returns the saturation of a normalised pixel."""
    if max(pixel) == min(pixel):
        return 0
    saturation = (max(pixel) - min(pixel)) / (max(pixel) + min(pixel))
    if luminance(pixel) > 0.5:
        saturation = -saturation / 2.0
    return saturation


def hue(pixel):
    """Returns the hue of a normalised pixel."""
    if saturation(pixel) == 0:
        return 0
    for i in range(3): # Offset is the indix of the most significant color
        if max(pixel) == pixel[i]:
            offset = i
            break
    hue = ((pixel[(offset + 1) % 3] - pixel[(offset + 2) % 3]) / (max(pixel) - min(pixel)) + 2 * offset) * 60
    return hue + 360 * (hue < 0)


def multPix(x, y):
    """Returns the product of two pixels."""
    return (x[0]*y[0], x[1]*y[1], x[2]*y[2])


def diff(x, y):
    """Returns the sum of all of an iterable's coefs difference."""
    return sum([abs(x[i] - y[i]) for i in range(len(x))])


def avg(lineData):
    """Returns the average color of a line."""
    pixDataSum = [0, 0, 0]
    for pixData in lineData:
        for x, chanData in enumerate(pixData):
            pixDataSum[x] += chanData
    for x, chanData in enumerate(pixDataSum):
        pixDataSum[x] = chanData/len(lineData)
    return pixDataSum


def randBool():
    """Returns True or False equiprobabily."""
    return random() > 0.5


def linInterp(x, xMin=0, xMax=1, yMin=0, yMax=1):
    """Simple linear interpolation.
    Maps a value from one range to another."""
    return (x - xMin) * (yMax - yMin) / (xMax - xMin) + yMin


def expInterp(x, xMin=0, xMax=1, yMin=0, yMax=1, expRange=1):
    """Exponential interpolation function.
    The interpolation tends to be linear when expRange tends to 0.
    In other words : the greater expRange is, the stronger the exponential effect is."""
    return linInterp(exp(linInterp(x, xMin, xMax, 0, expRange)) - 1, 0, exp(expRange)-1, yMin, yMax)


def computeLine(data, width, y, threshold=None, alternativeThreshold=False, alternativeReverse=False, maxChunkSizeRange=None, ponderation=None, key=sum):
    """Pixel computation logic. Not really to be called manually..."""

    lineData = data[y*width : (y+1)*(width)]

    # If there's a significant ponderation, we modify the key callable to take care of it.
    if ponderation != None and max(ponderation) != min(ponderation):
        pondKey = lambda px: key(multPix(px, ponderation))
    else:
        pondKey = key

    if threshold == None:
        lineData.sort(key=pondKey)
        return lineData

    else:
        # Chunk limits computation
        chunksLimits = [0]
        for x, pixData in enumerate(lineData[1:-2]):
            if diff(pixData, lineData[x-1]) > threshold:
                if alternativeThreshold:
                    if chunksLimits == [0]:
                        transitionTriggered = False
                    if not transitionTriggered:
                        chunksLimits.append(x)
                    transitionTriggered = not transitionTriggered

                else:
                    chunksLimits.append(x)

                # Chunk splitting according to maxChunkSizeRange
                if maxChunkSizeRange != None:
                    lowBoundary = chunksLimits[-2]

                    chunkId = -1
                    while chunksLimits[chunkId] > lowBoundary + max(maxChunkSizeRange): #DOES.NOT.FUCKING.WORKS.
                          chunksLimits.insert(chunkId, chunksLimits[chunkId] - randint(*maxChunkSizeRange))
                          chunkId -= 1

        chunksLimits.append(len(lineData))

        # Chunk data computation
        lineDataOut = []
        for chunkId, chunkLimit in enumerate(chunksLimits[:-1]):
            chunkData = lineData[chunkLimit:chunksLimits[chunkId+1]]
            chunkData.sort(key=pondKey)
            if alternativeReverse and chunkId % 2 == 0:
                chunkData.reverse()
            lineDataOut += chunkData
        return lineDataOut


def sort(image, transpose=False, name='generated', **kwargs):
    """Outputs a sorted version of an image to disk.\n
    :param image: The image to process.
    :param ponderation: Containing three ponderation values (r, g, b). None is faster.
    :param key: Callable that will be used as sorting key.
    :param transpose: If True, columns are sorted instead of lines.
    :param threshold: Cluster detection threshold.
    :param alternativeThreshold: If True, only one chunk edge in two triggers.
    :param alternativeReverse: If True, reverses sorted chunk one time in two.
    :param maxChunkSizeRange: The max chunk size will be randomly picked between (min, max) if provided.
    :param name: The output file name, without extension.

    :type image: PIL.Image
    :type ponderation: tuple / NoneType
    :type key: function
    :type transpose: bool
    :type threshold: int / NoneType
    :type alternativeThreshold: int
    :type alternativeReverse: bool
    :type maxChunkSizeRange: tuple / NoneType
    :type name: str"""

    if transpose:
        image = image.transpose(Image.ROTATE_90)
    data = list(image.getdata())

    print("=== Starting processing ===")
    timer = clock()
    imageData = []

    print("Now computing...")
    for y in range(0, image.height):
        imageData.append(computeLine(data, image.width, y, **kwargs))
    imageData = tuple(chain(*imageData))

    print("Saving %s ..." % name)
    image.putdata(imageData)
    if transpose:
        image = image.transpose(Image.ROTATE_270)
    image.save(name + '.png')

    # TODO: Logging

    print("Done (%ss)." % str(clock()-timer))
    return True


if __name__ ==  '__main__':
    # f = input("Path: ")
    # if f == '':
    #     exit()

    # images = [Image.open(f) for f in listdir('.') if f.endswith('jpg')]

    f = "d.jpg"
    image = Image.open(f)


    # args = []
    # for transpose in (True, False):
    #     for threshold in (64, 128, 256):
    #         for alternativeReverse in (True, False):
    #             for ponderation in (ALL, MAX):
    #                 args.append((image, ponderation, transpose, threshold, False, alternativeReverse, None, f+str(transpose)+str(ponderation)+str(threshold)+str(alternativeReverse)))

    # args = [(image, ALL, True, 150, False, True, None, format(i, '05d')) for i, image in enumerate(images)]

    kwargs = [{
        'image': image,
        'key': lambda px: hue(normalize(px)),
        'ponderation': (random()-.5, random()-.5, random()-.5),
        'transpose': randBool(),
        'threshold': randint(0, 400),
        'alternativeThreshold': randBool(),
        'alternativeReverse': randBool(),
        'maxChunkSizeRange': None,
        'name': format(i, '04d')
    } for i in range(8)]

    # nbImgs = 300
    # args = [(image, (linInterp(i, 0, nbImgs, -1, 1), linInterp(i, 0, nbImgs, 1, -1), 0), False, 100, False, False, None, format(i, '04d')) for i in range(nbImgs)]


    pool = Pool(8)
    pool.map(lambda dic: sort(**dic), kwargs)
    pool.close()
    pool.terminate()
    exit()
