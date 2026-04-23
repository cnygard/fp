import numpy as np

def spiral(center,size):
    order = np.empty(size**2)
    row,col = center
    dirBorder = [
        col+1, # right
        row+1, # down
        col-1, # left
        row-1  # up
    ]
    dirIdx = 0
    i = 0
    while col < size and col >= 0 and row < size and row >= 0:
        # print(f'({row},{col})')
        order[i] = col + row*size
        i += 1
        if dirIdx == 0:
            col += 1
            d = col
        elif dirIdx == 1:
            row += 1
            d = row
        elif dirIdx == 2:
            col -= 1
            d = col
        elif dirIdx == 3:
            row -= 1
            d = row
        if d == dirBorder[dirIdx]:
            if dirIdx > 1:
                dirBorder[dirIdx] -= 1
            else:
                dirBorder[dirIdx] += 1
            dirIdx = (dirIdx + 1) % 4
    return order

flat = np.empty(225)
c = 0
for i in spiral((7,7),15):
    flat[int(i)] = c
    c += 1

flat.reshape((15,15))
for i in range(225):
    print(f'{int(flat[i]):6}',end=' ')
    if i % 15 == 14:
        print("")
