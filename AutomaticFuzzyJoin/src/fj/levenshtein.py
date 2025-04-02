'''implementation of Levenshtein distance algorithm'''


def levenshtein_distance(str_1:str, str_2:str) -> int:
    '''Function to compute the Levenshtein distance between two strings'''

    m = len(str_1)
    n = len(str_2)

    # Initialize a matrix to store the edit distances
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j


    # Fill the matrix using dynamic programming to compute edit distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str_1[i - 1] == str_2[j - 1]:
                # Characters match, good! No extra cost
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Characters don't match, choose minimum cost
                # among insertion, deletion, or substitution
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])


    # Print the matrix
    for row in dp:
        print(row)

    # Return the edit distance between the strings
    return dp[m][n]

if __name__ == "__main__":
    # Driver code
    str_a = "kitten"
    str_b = "sitting"

    # Function Call
    distance = levenshtein_distance(str_a, str_b)
    print(f"Levenshtein Distance: {distance}")
    