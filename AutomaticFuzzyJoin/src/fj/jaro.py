s_1 = "helpl"
s_2 = "hellp"

len_s1 = len(s_1)
len_s2 = len(s_2)

matches = 0

for i in range(len_s1):
    for j in range(len_s2):
        if s_1[i] == s_2[j]:
            matches += 1


transpositions = matches // 2

similarity = ((matches / len_s1) + (matches / len_s2)    + ((matches - transpositions) / matches)) / 3

print(similarity)