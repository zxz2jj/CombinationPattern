list1 = [[1, 2], [2, 2], [1, 2]]

temp = []
for i in list1:
    if i not in temp:
        temp.append(i)
print(temp)