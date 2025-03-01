class NQueensSolver:
    def __init__(self, N):
        self.N = N
        self.board = [-1] * N  # board[i] stores the column position of the queen in row i
        self.solutions = []

    # Person 2: Check if a queen can be placed in (row, col)
    def is_safe(self, row, col):
        for prev_row in range(row):
            prev_col = self.board[prev_row]
            # Check same column or diagonals
            if prev_col == col or abs(prev_col - col) == abs(prev_row - row):
                return False
        return True

    # Person 1: Recursive backtracking function
    def solve(self, row=0):
        if row == self.N:  # Base case: all queens are placed
            self.solutions.append(self.board[:])  # Store the solution
            return
        
        for col in range(self.N):  # Try placing queen in each column
            if self.is_safe(row, col):
                self.board[row] = col
                self.solve(row + 1)  # Recur for the next row
                self.board[row] = -1  # Backtrack

    # Person 3: Display solutions
    def print_solutions(self):
        for sol in self.solutions:
            for row in range(self.N):
                line = ['.'] * self.N
                line[sol[row]] = 'Q'
                print(" ".join(line))
            print("\n" + "-" * (2 * self.N - 1))  # Separator

# Person 4: Testing & Debugging
if __name__ == "__main__":
    N = 8  # Change for different board sizes
    solver = NQueensSolver(N)
    solver.solve()
    solver.print_solutions()
    print(f"Total solutions found: {len(solver.solutions)}")
