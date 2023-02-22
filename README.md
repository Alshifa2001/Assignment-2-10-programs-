# Assignment-2-10-programs-
***********valid anagram*************
def are_anagram(s1, s2):

    if len(s1) != len(s2):
        return False

    if (sorted(s1) == sorted(s2)):
        print("The strings are anagrams.")
    else:
        print("The strings aren't anagrams.")

s1 = "listen"
s2 = "danger"
are_anagram(s1, s2)

*********first and last index************
def findFirstAndLast(arr, n, x):
    first = -1
    last = -1
    for i in range(0, n):
        if (x != arr[i]):
            continue
        if (first == -1):
            first = i
        last = i
 
    if (first != -1):
        print("First Occurrence = ", first,
              " \nLast Occurrence = ", last)
    else:
        print("Not Found")
 
 
# Driver code
arr = [1, 2, 2, 2, 2, 3, 4, 7, 8, 8]
n = len(arr)
x = 8
findFirstAndLast(arr, n, x)

******************Kth largest************
class Solution(object):
   def findKthLargest(self, nums, k):
      nums.sort()
      if k ==1:
         return nums[-1]
      temp = 1
      return nums[len(nums)-k]
ob1 = Solution()
print(ob1.findKthLargest([56,14,7,98,32,12,11,50,45,78,7,5,69], 5))

************symmetric tree*********
class Node:
    def _init_(self, key):
        self.key = key
        self.left = None
        self.right = None

def are_symmetric(root1, root2):
    if root1 is None and root2 is None:
        return True
    if(root1 is  not None and root2 is not None):
        if root1.key == root2.key:
            return are_symmetric(root1.left, root2.right) and are_symmetric(root1.right, root2.left)
    return False

def is_symmetric(root):
    
    return are_symmetric(root, root)


root = Node(1)
root.left = Node(2)
root.right = Node(2)
root.left.left = Node(3)
root.left.right = Node(4)
root.right.left = Node(4)
root.right.right = Node(3)
print ("Symmetric" if is_symmetric(root) == True else "Not symmetric")

******************generate parenthesis************
def printParenthesis(str, n):
	if(n > 0):
		_printParenthesis(str, 0,
						n, 0, 0)
	return


def _printParenthesis(str, pos, n,
					open, close):

	if(close == n):
		for i in str:
			print(i, end="")
		print()
		return
	else:
		if(open > close):
			str[pos] = '}'
			_printParenthesis(str, pos + 1, n,
							open, close + 1)
		if(open < n):
			str[pos] = '{'
			_printParenthesis(str, pos + 1, n,
							open + 1, close)
n = 3
str = [""] * 2 * n
printParenthesis(str, n)

*************gas station******************
gas=[1,2,3,4,5]
cost=[3,4,5,1,2]
tank,shortage,start=0,0,0
for i in range(len(gas)):
    tank+=gas[i]
    if tank>=cost[i]:
        tank-=cost[i]
    else:
        shortage+=cost[i]-tank
        start=i+1
        tank=0
if start==len(gas) or tank<shortage:
    print(-1) 
else:
    print(start)                 

***********course schedule*************
class Solution(object):
   def canFinish(self, numCourses, prerequisites):
      if len(prerequisites) == 0:
         return True
      visited = [0 for i in range(numCourses)]
      adj_list = self.make_graph(prerequisites)
      for i in range(numCourses):
         if not visited[i]:
            if not self.cycle(adj_list,visited,i):
               return False
      return True
   def cycle(self,adj_list,visited,current_node = 0):
      if visited[current_node] ==-1:
         return False
      if visited[current_node] == 1:
         return True
      visited[current_node] = -1
      if(current_node in adj_list):
         for i in adj_list[current_node]:
            if not self.cycle(adj_list,visited,i):
               return False
      visited[current_node] = 1
      return True
   def make_graph(self,array):
      adj_list = {}
      for i in array:
         if i[1] in adj_list:
            adj_list[i[1]].append(i[0])
         else:
            adj_list[i[1]] = [i[0]]
      return adj_list
ob = Solution()
print(ob.canFinish(2, [[1,0]]))

***********kth permutation**************
def generate_permutations(ch, idx, result):
	# base case
	if idx == len(ch):
		str1 = ""
		result.append(str1.join(ch))
		return

	for i in range(idx, len(ch)):
		ch[i], ch[idx] = ch[idx], ch[i]
		generate_permutations(ch, idx + 1, result)
		ch[i], ch[idx] = ch[idx], ch[i]

def findKthPermutation(n, k):
	s = ""
	result = []

	for i in range(1, n + 1):
		s += str(i)

	ch = [*s]
	generate_permutations(ch, 0, result)


	result.sort()

	return result[k - 1]


if __name__ == "__main__":
	n = 3
	k = 4

	kth_perm_seq = findKthPermutation(n, k)
	print(kth_perm_seq)

************minimum window substring**********
from collections import Counter

def contains_all(freq1, freq2):
    for ch in freq2:
        if freq1[ch] < freq2[ch]:
            return False
    return True


def min_window(s, t):
    n, m = len(s), len(t)
    if m > n or m == 0:
        return ""
    freqt = Counter(t)
    shortest = " "*(n+1)
    for length in range(1, n+1):
        for i in range(n-length+1):
            sub = s[i:i+length]
            freqs = Counter(sub)
            if contains_all(freqs, freqt) and length < len(shortest):
                shortest = sub
    return shortest if len(shortest) <= n else ""


def min_window(s, t):
    n, m = len(s), len(t)
    if m > n or t == "":
        return ""
    freqt = Counter(t)
    start, end = 0, n+1
    for length in range(1, n+1):
        freqs = Counter()
        satisfied = 0
        for ch in s[:length]:
            freqs[ch] += 1
            if ch in freqt and freqs[ch] == freqt[ch]:
                satisfied += 1
        if satisfied == len(freqt) and length < end-start:
            start, end = 0, length
        for i in range(1, n-length+1):
            freqs[s[i+length-1]] += 1
            if s[i+length-1] in freqt and freqs[s[i+length-1]] == freqt[s[i+length-1]]:
                satisfied += 1
            if s[i-1] in freqt and freqs[s[i-1]] == freqt[s[i-1]]:
                satisfied -= 1
            freqs[s[i-1]] -= 1
            if satisfied == len(freqt) and length < end-start:
                start, end = i, i+length
    return s[start:end] if end-start <= n else ""


def min_window(s, t):
    n, m = len(s), len(t)
    if m > n or t == "":
        return ""
    freqt = Counter(t)
    start, end = 0, n
    satisfied = 0
    freqs = Counter()
    left = 0
    for right in range(n):
        freqs[s[right]] += 1
        if s[right] in freqt and freqs[s[right]] == freqt[s[right]]:
            satisfied += 1
        if satisfied == len(freqt):
            while s[left] not in freqt or freqs[s[left]] > freqt[s[left]]:
                freqs[s[left]] -= 1
                left += 1
            if right-left+1 < end-start+1:
                start, end = left, right
    return s[start:end+1] if end-start+1 <= n else ""


if __name__ == "__main__":
    s = "ABCFEBECEABEBAABACEDBCDEBACE"
    t= "ABCA"

    print(min_window(s, t))

************largest rectangle in histogram**********
def largest_rectangle(heights):
    max_area = 0
    for i in range(len(heights)):
        left = i
        while left - 1 >= 0 and heights[left - 1] >= heights[i]:
            left -= 1
        right = i
        while right + 1 < len(heights) and heights[right + 1] >= heights[i]:
            right += 1
        max_area = max(max_area, heights[i] * (right - left + 1))
    return max_area


def rec(heights, low, high):
    if low > high:
        return 0
    elif low == high:
        return heights[low]
    else:
        minh = min(heights[low:high + 1])
        pos_min = heights.index(minh, low, high + 1)
        from_left = rec(heights, low, pos_min - 1)
        from_right = rec(heights, pos_min + 1, high)
        return max(from_left, from_right, minh * (high - low + 1))


def largest_rectangle(heights):
    return rec(heights, 0, len(heights) - 1)


def largest_rectangle(heights):
    heights = [-1] + heights + [-1]
    from_left = [0] * len(heights)
    stack = [0]
    for i in range(1, len(heights) - 1):
        while heights[stack[-1]] >= heights[i]:
            stack.pop()
        from_left[i] = stack[-1]
        stack.append(i)
    from_right = [0] * len(heights)
    stack = [len(heights) - 1]
    for i in range(1, len(heights) - 1)[::-1]:
        while heights[stack[-1]] >= heights[i]:
            stack.pop()
        from_right[i] = stack[-1]
        stack.append(i)
    max_area = 0
    for i in range(1, len(heights) - 1):
        max_area = max(max_area, heights[i] * (from_right[i] - from_left[i] - 1))
    return max_area


def largest_rectangle(heights):
    heights = [-1] + heights + [-1]
    max_area = 0
    stack = [(0, -1)]
    for i in range(1, len(heights)):
        start = i
        while stack[-1][1] > heights[i]:
            top_index, top_height = stack.pop()
            max_area = max(max_area, top_height * (i - top_index))
            start = top_index
        stack.append((start, heights[i]))
    return max_area


if __name__ == '__main__':
    hist = [6, 2, 5, 4, 5, 1, 6]

    # Function call
    print("Maximum area is",
         largest_rectangle(hist))
