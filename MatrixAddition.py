class mylist1:
    def __init__(self,n,m):
        self.__list = []
        self.sizerow = n
        self.sizecol = m
        for i in range(n):
            l = []
            for j in range(m):
                l.append(0)
            self.__list.append(l)

    def changesize(self):
        self.sizerow = len(self.__list)
        self.sizecol = len(self.__list[0])

    def set(self, i, j, x):
        self.__list[i][j] = x

    def show(self):
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                print(self.__list[i][j], end = '\t')
        print(' ')

    def __add__(self, m2):
        l = mylist1(self.sizerow, self.sizecol)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                l.set(i, j, self.__list[i][j] + m2.__list[i][j])
        return l

A= mylist1(5,4)
A.set(2, 3, 4)
A.set(3, 2, 7)
print('D:')
print(type(A))
A.show()

print('B:')
B = mylist1(5,4)
B.set(1, 2, 3)
B.set(3, 2, 9)
print(type(B))
B.show()

print('C:')
C = A + B
print(type(C))
C.show()
