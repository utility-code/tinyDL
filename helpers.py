def pbar(iterable, length=50, prefix='', suffix='', decimals=1, fill='█', printEnd="\r"):
    total = len(iterable)

    def printPbar(iteration):
        percent = ("{0:." + str(decimals)+"f}").format(100 *
                                                       (iteration/float(total)))
        filledLength = int(length*iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

    printPbar(0)
    for i, item in enumerate(iterable):
        yield item
        printPbar(i+1)

    print()

#  items = list(range(0, 57))
#  import time
#  for i in pbar(items):
    #  time.sleep(0.1)
