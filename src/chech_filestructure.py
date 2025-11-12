import os

def tree(path, level=0, max_level=2):
    if level > max_level:
        return
    for name in os.listdir(path):
        p = os.path.join(path, name)
        print(" " * (level * 2) + "|-- " + name)
        if os.path.isdir(p):
            tree(p, level + 1, max_level)

tree(".", 0, 3)
