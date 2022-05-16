
start = 3
end = 5
ans_len = end - start

a = ['i', 'am', 'a', 'big', 'red', 'teapot', 'for', 'drinks']
a.insert(start, "<b>")
a.insert(end + ans_len + 1, "</b>")
print(a)
