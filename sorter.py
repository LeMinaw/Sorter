from PIL             import Image, ImageDraw, ImageColor
from bitstring       import BitArray
from math            import ceil, sqrt, exp
from multiprocessing import Pool
from itertools       import chain
from time            import clock
from random          import random, randint
from os              import close, listdir
from copy            import deepcopy

import cProfile


(MAX, ALL, AUTO, DISABLED) = ('max', 'all', 'auto', 'disabled')
K = (0, 0, 0)
R = (1, 0, 0)
G = (0, 1, 0)
B = (0, 0, 1)
C = (0, 1, 1)
M = (1, 0, 1)
Y = (1, 1, 0)
W = (1, 1, 1)


def randstr(size=16):
    chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
             'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    string = ""
    for i in range(16):
        string += chars[randint(0, len(chars)-1)]
    return string


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


def ponderate(coefs):
    """Takes sub-weights in params, and applies sub-weights to compute a final pixel weight."""
    return lambda x: x[0]*coefs[0] + x[1]*coefs[1] + x[2]*coefs[2] #SpeedHackOfTheDeath
    return lambda x: sum(multPix(x, coefs))


def computeLine(data, width, threshold, alternativeThreshold, alternativeReverse, maxChunkSizeRange, ponderation, y):
    """Pixel computation logic. Not really to be called manually..."""

    lineData = data[y*width : (y+1)*(width)]

    if ponderation == MAX:
        sortFunction = max
    elif ponderation == ALL:
        sortFunction = sum
    elif ponderation == AUTO:
        sortFunction = ponderate(avg(lineData))
    else:
        sortFunction = ponderate(ponderation)

    if threshold == DISABLED:
        lineData.sort(key=sortFunction)
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
            chunkData.sort(key=sortFunction)
            if alternativeReverse and chunkId % 2 == 0:
                chunkData.reverse()
            lineDataOut += chunkData
        return lineDataOut


def sort(image, ponderation=ALL, transpose=False, threshold=DISABLED, alternativeThreshold=False, alternativeReverse=False, maxChunkSizeRange=None, name='generated'):
    """Returns a sorted version of an image.\n
    image: PIL.Image, the image to process.
    ponderation: A tuple of three ponderation values. Can also be MAX or ALL.
    transpose: Bool, if true, columns are sorted instead of lines.
    name: The output file name."""

    if transpose:
        image = image.transpose(Image.ROTATE_90)
    data = list(image.getdata())

    print("=== Starting processing ===")
    timer = clock()
    imageData = []

    print("Now computing...")
    for y in range(0, image.height):
        imageData.append(computeLine(data, image.width, threshold, alternativeThreshold, alternativeReverse, maxChunkSizeRange, ponderation, y))
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

    f = "o.jpg"
    image = Image.open(f)


    # args = []
    # for transpose in (True, False):
    #     for threshold in (64, 128, 256):
    #         for alternativeReverse in (True, False):
    #             for ponderation in (ALL, MAX):
    #                 args.append((image, ponderation, transpose, threshold, False, alternativeReverse, None, f+str(transpose)+str(ponderation)+str(threshold)+str(alternativeReverse)))

    # args = [(image, ALL, True, 150, False, True, None, format(i, '05d')) for i, image in enumerate(images)]

    args = [(image, (random()-.5, random()-.5, random()-.5), randBool(), randint(0, 400), randBool(), randBool(), None, format(i, '04d')) for i in range(10)]

    # nbImgs = 300
    # args = [(image, MAX, False, expInterp(i, 0, nbImgs, 0, 512, 1), False, False, None, format(i, '04d')) for i in range(nbImgs)]

    # args = [(image, MAX, False, , False, False, None, "image"]


    pool = Pool(8)
    pool.starmap(sort, args)
    pool.close()
    pool.terminate()
    exit()


    # cProfile.run('sort(image, ponderation=B, transpose=True, threshold=150, alternativeThreshold=False, name="test")')
    # exit()
