import random
from user import solve
alpha = [[[0,0,0,0,8,8,8,8],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]],

       [[0,0,0,0,8,8,8,8],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0]],

        [[0,0,0,0,8,8,8,8],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,8,8,8]],

        [[0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,0,0,0],
        [0,0,0,0,8,8,8,8]],

        [[8,8,8,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]],

        [[8,8,8,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0]],

        [[8,8,8,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [8,8,8,8,0,0,0,0]],

        [[0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [0,0,0,8,0,0,0,0],
        [8,8,8,8,0,0,0,0]],

        [[0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,8,0,0,8,0,0],
        [0,0,8,0,0,8,0,0],
        [0,0,8,0,0,8,0,0],
        [0,0,8,8,8,8,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]],

        [[0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,8,8,8,8,0,0],
        [0,0,8,0,0,8,0,0],
        [0,0,8,0,0,8,0,0],
        [0,0,8,0,0,8,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]],]

photo=[[0 for j in range(64)] for i in range(16)]

def init():
    for i in range(16):
        for j in range(64):
            photo[i][j]=0

def bitblt(x ,y ,type):
    for i in range(8):
        for j in range(8):
            photo[i+y][j+x]=alpha[type][i][j]

def run(n):
    cur=0
    r=1000
    for i in range(4):
        cur+=random.randrange(0,9)
        y=random.randrange(0,9)
        type=n//r%10
        bitblt(cur,y,type)
        cur+=8
        r//=10


    for i in range(16):
        for j in range(64):
            if random.randrange(0,10)==0:
                photo[i][j] = (8 if photo[i][j]==0 else 0)


if __name__ == "__main__":
    result=0
    for i in range(10):
        number=random.randrange(0,10000)
        init()
        run(number)
        if number == solve(photo):
            result+=1
    print("result : ",result)