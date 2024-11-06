# Function to print the board
def print_board(board):
    for row in board:
        print(" ".join(str(cell) for cell in row))
    print("\n")

# Function to check if a queen can be placed at board[row][col]
def is_safe(board, row, col, n):
    # Check the column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper-left diagonal
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper-right diagonal
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

# Function to solve the N-Queens problem
def solve_n_queens(board, row, n):
    # Base case: If all queens are placed
    if row == n:
        print_board(board)
        return True

    # Place queen in each column in the current row
    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1  # Place queen

            # Recur to place the rest of the queens
            if solve_n_queens(board, row + 1, n):
                return True

            # If placing queen in board[row][col] doesn't lead to a solution
            # then backtrack
            board[row][col] = 0  # Remove queen

    return False

# Initialize the board and solve the 8-Queens problem
n = 8
board = [[0 for _ in range(n)] for _ in range(n)]

if not solve_n_queens(board, 0, n):
    print("No solution exists")
