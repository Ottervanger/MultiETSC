#!/usr/bin/env python
import random

"""
Adopted from Orange3 (https://github.com/biolab/orange3) which is published
under the [GPL-3.0]+ license.

[GPL-3.0]: https://www.gnu.org/licenses/gpl-3.0.en.html

Changes by Gilles Ottervanger:
    Some refactoring
    Numerical labels indicating rank means
"""

def cdGraph(avranks, names, cd, filename,
            width=7, textspace=1.7, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Requires:
        numpy
        matplotlib
        math
    """
    try:
        import math
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib and numpy.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    tempsort = sorted([(a, i) for i, a in enumerate(avranks)], reverse=True)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    lowv = 1

    cline = 0.4

    k = len(avranks)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        return textspace + scalewidth / (k - lowv) * (k - rank)

    distanceh = 0.25

    if cd:
        # get pairs of non significant methods

        def get_lines(sums, hsd):
            lines = []
            j = -1
            for i in range(len(sums)):
                pj = j
                j = max(i+1, j)
                while j < len(sums) and abs(sums[i] - sums[j]) <= hsd:
                    j += 1
                if i < j-1 and j > pj:
                    lines += [(i, j-1)]
            return lines

        lines = get_lines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

        # add scale
        distanceh = 0.25
        cline += distanceh

    # calculate height needed height of an image
    vspace = 0.3
    minnotsignificant = max(2 * vspace, linesblank)
    height = cline + ((k + 1) / 2) * vspace + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    def lmul(l, f):
        return [a * f for a in l]


    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05
    smalltext = '7'

    tick = None
    for a in list(np.arange(lowv, k, 0.5)) + [k]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, k + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * vspace
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")
        text(textspace - 0.2, chei + 0.12, f'{ssums[i]:.3f}',
             ha="right", va="center", fontsize=smalltext)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * vspace
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i], ha="left", va="center")
        text(textspace + scalewidth + 0.2, chei + 0.12, f'{ssums[i]:.3f}',
             ha="left", va="center", fontsize=smalltext)

    # upper scale
    def drawScale():
        begin, end = rankpos(k), rankpos(k - cd)
        args = dict(color='k', linewidth=0.7)
        bw, ew = begin/width, end/width
        distancehh = distanceh / height
        bt = bigtick / (2 * height)
        ax.plot([bw, ew], [distancehh, distancehh], **args)
        ax.plot([bw, bw], [distancehh + bt, distancehh - bt], **args)
        ax.plot([ew, ew], [distancehh + bt, distancehh - bt], **args)
        ax.text((ew+bw)/2, distancehh - 0.05/height, 'CD', ha="center", va="bottom", **kwargs)

    # no-significance lines
    def drawLines(lines, side=0.02, height=0.1):
        start = cline + 0.2
        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=2.5)
            start += height

    drawScale()
    drawLines(lines)

    if filename:
        print_figure(fig, filename, **kwargs)
    fig.clear()


if __name__ == '__main__':
    n = 11
    random.seed(0)
    ssums = sorted([i+random.gauss(1, .5) for i in range(n)])
    cd = 1.5
    filename = 'cd.pdf'

    print(ssums)
    cdGraph(ssums, [f'method {i:2d}' for i in range(n)], cd, filename)
