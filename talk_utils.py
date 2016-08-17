import matplotlib.pyplot as plt


def plotter(line_x, line_y, scatter_x, scatter_y):
    _, ax = plt.subplots()
    ax.plot(line_x, line_y, color='red')
    ax.scatter(scatter_x, scatter_y)


class Matrix(object):

    def __init__(self, vals):
        self.nrows, self.ncols = len(vals), len(vals[0])
        self.dim = self.nrows, self.ncols
        for row in vals:
            assert len(row) == self.ncols
        self.vals = vals

    def __repr__(self):
        return '\n'.join(' '.join(map(str, row)) for row in self.vals)

    def __mul__(self, other):
        assert self.ncols == other.nrows, "Cannot multiply {}-matrix by {}-matrix".format(self.dim, other.dim)
        result_vals = []
        for row_idx in range(self.nrows):
            result_vals.append([0] * other.ncols)
            for col_idx in range(other.ncols):
                result_vals[row_idx][col_idx] = sum(a * b for a, b in zip(self.row(row_idx), other.col(col_idx)))
        return Matrix(result_vals)

    def __rmul__(self, other):
        return Matrix([[other * j for j in row] for row in self.vals])

    def row(self, idx):
        return self.vals[idx]

    def col(self, idx):
        return [row[idx] for row in self.vals]

    def _copy_vals(self):
        return [row.copy() for row in self.vals]

    @property
    def T(self):
        return Matrix(list(zip(*self.vals)))

    def inverse(self):
        """Super optimistic, non-error tolerant gaussian elimination."""
        assert self.nrows == self.ncols
        vals = self._copy_vals()
        inverse = eye(self.nrows)
        for diag in range(self.nrows):
            pivot = vals[diag][diag]
            for col in range(self.ncols):  # Multiply row to make pivot 1
                vals[diag][col] /= pivot
                inverse[diag][col] /= pivot
            for row_idx in range(diag + 1, self.nrows):  # eliminate rows below
                val = vals[row_idx][diag]
                for col_idx in range(self.ncols):
                    vals[row_idx][col_idx] += -val * vals[diag][col_idx]
                    inverse[row_idx][col_idx] += -val * inverse[diag][col_idx]

        for diag in range(self.nrows - 1, -1, -1):  # back substitute
            for row in range(diag - 1, -1, -1):
                mult = -vals[row][diag]
                for col in range(self.ncols):
                    vals[row][col] += mult * vals[diag][col]
                    inverse[row][col] += mult * inverse[diag][col]
        return Matrix(inverse)


def eye(rows):
    identity = [[0] * rows for _ in range(rows)]
    for row in range(rows):
        identity[row][row] = 1
    return identity
