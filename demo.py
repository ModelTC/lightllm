from typing import List

class Solution:
    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        num_to_count = [0 for _ in range(10)]
        for e in digits:
            num_to_count[e] += 1
        self.ans_list = [0]
        self.num_count = len(digits)

        self.dfs(num_to_count, 0, 0)
        return self.ans_list[1:]

    
    def dfs(self, num_to_count:List[int], loc:int, num):
        if loc == 3:
            if num % 2 == 0 and self.ans_list[-1] != num:
                self.ans_list.append(num)
                return
        
        if loc == 0:
            for i in range(1, 10):
                if num_to_count[i] > 0:
                    num_to_count[i] -= 1
                    new_num = num * 10 + i
                    self.dfs(num_to_count, loc + 1, new_num)
                    num_to_count[i] += 1
        if loc == 1:
            for i in range(0, 10):
                if num_to_count[i] > 0:
                    num_to_count[i] -= 1
                    new_num = num * 10 + i
                    self.dfs(num_to_count, loc + 1, new_num)
                    num_to_count[i] += 1
        
        if loc == 2:
            for i in range(0, 10, 2):
                if num_to_count[i] > 0:
                    num_to_count[i] -= 1
                    new_num = num * 10 + i
                    self.dfs(num_to_count, loc + 1, new_num)
                    num_to_count[i] += 1


