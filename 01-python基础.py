-*- coding: utf-8 -*-
# 练习一
nums = [1, 3, 5, 7, 9]
new_nums = []
for num in nums:
    new_num = num + 2
    new_nums.append(new_num)
print(new_nums)

# 练习二创建字典
info = {"name": "柳佳慧", 
        "age": 20, 
        "major": "AI"}

## print(info.name)是错误的，正确调用方法：
print(info["name"])
# 添加键值对不会,-------修正
info["city"]="北京"
# 遍历打印不会-------修正
for key,value in info.items():
    print(key,":",value)

# 练习三
def calc(a,b):
    return a*b+a-b

num3=calc(3,4)
print(num3)

for i in range(1,20):
    if i%2==0:
        print(i,"是偶数")

# 练习五
class Model:
    def __init__(self,name):
        self.name=name
        
    def show_info(self):
        print("模型名称："+self.name)
        
m=Model("我的神经网络")
m.show_info()