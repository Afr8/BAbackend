# This is a sample Python script.
import Edgedetection
import Gates
import numpy as np

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
from collections import Counter
import random
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Input1 = Gates.Input()
    # Input2 = Gates.Input()
    # And1 = Gates.And([Input1, Input2])
    # Output = Gates.Output(And1)
    # Input1.update
    # Input2.update
    # Input1.switch()
    # Input2.switch()
    # print(Output.getValue())
    # boxes = np.zeros((7,5))
    #boxes= [[1,2],[3,4],[5,6],[7,8]]
    # print(boxes)
    #boxes = np.array(boxes)
    # pixels = np.zeros((500,500))
    # #x,y = 20
    # radiust =2
    # pixels[(2,2)] = 1
    # for i in range(-radiust, +radiust + 1):
    #     for j in range(-radiust, +radiust + 1):
    #         if ( radiust > i and i > -radiust) and (radiust > j and j > -radiust):
    #             continue
    #         pixels[(i+2,j+2)] = 1
    # print("Done")
    import numpy as np
    Colors =[]
    counterc = 0

    r = lambda: random.randint(0, 4) * 64
    while len(Colors) < 100:
        curColor =(r(), r(), r())
        if sum(list(curColor)) >= 641:
            continue
        if curColor in Colors:
            continue
        Colors.append(curColor)

    #
    # integers = [2, 4, 6, 8, 10, 3]
    # liste = [i % 2 == 0 for i in integers]
    # c = Counter(liste)
    # print(c)
    # # print(integers[np.where(list == c[1])])
    r=0
    # for i in range(0,10):
    #     r = lambda: random.randint(0, 25)*10
    #
    #     print('#%02X%02X%02X' % (r(), r(), r()))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
