from random import random, randint

num = randint(1, 100)
num_input = eval(input('请输入一个数字：'))
while num != num_input:
    if num < num_input:
        print('数字太大了')
    else:
        print('数字太小了')
    num_input = eval(input('请输入一个数字：'))
print('恭喜你，猜对了！')