Module tinydl.helpers
=====================

Functions
---------

    
`checkifdir(logdir='./experiments/')`
:   [summary]
    
    Args:
        logdir ([type], optional): [description]. Defaults to logdir.
    Check if the directory exists

    
`getexpno(logdir='./experiments/')`
:   [summary]
    
    Args:
        logdir ([type], optional): [description]. Defaults to logdir.
    
    Returns:
        [type]: [description]
    Return the current experiment number

    
`info(arr, n='', p=0)`
:   [summary]
    
    Args:
        arr ([type]): [description]
        n (str, optional): [description]. Defaults to "".
        p (int, optional): [description]. Defaults to 0.
    Give an array, get a description. Shape, count, mean etc.

    
`pbar(iterable, length=50, listofextra=[], prefix='', suffix='', decimals=1, fill='█', printEnd='\r')`
:   [summary]
    
    Args:
        iterable ([type]): [description]
        length (int, optional): [description]. Defaults to 50. Change if you have a big/small screen.
        listofextra (list, optional): [description]. Defaults to []. Extra text
        prefix (str, optional): [description]. Defaults to "".
        suffix (str, optional): [description]. Defaults to "".
        decimals (int, optional): [description]. Defaults to 1.
        fill (str, optional): [description]. Defaults to "█". Change if you want a different block
        printEnd (str, optional): [description]. Defaults to "".
    
    Yields:
        [type]: [description]
    
    Custom progress bar.

    
`savemodel(model, total_loss)`
:   [summary]
    
    Args:
        model ([type]): [description]
        total_loss ([type]): [description]
    Save the model to a pickle file