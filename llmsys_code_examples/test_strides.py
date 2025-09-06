class ArrayIndexingDemo:
    """
    A comprehensive class to understand array indexing concepts:
    1. Row-major format (C-style)
    2. Strides-based indexing
    3. Conversion between 1D flat data and 2D array representations
    """
    
    def __init__(self, data, shape):
        """
        Initialize the array with flat data and shape.
        
        Args:
            data (list): Flat 1D array data
            shape (tuple): Shape as (rows, cols)
        """
        self.data = data
        self.rows, self.cols = shape
        self.shape = shape
        
        # Validate that data size matches shape
        if len(data) != self.rows * self.cols:
            raise ValueError(f"Data size {len(data)} doesn't match shape {shape}")
        
        # Calculate strides for row-major format
        # Stride[i] = number of elements to skip to move one step in dimension i
        self.strides = (self.cols, 1)  # (col_stride, element_stride)
        
        print(f"Initialized array with shape {shape}")
        print(f"Data: {data}")
        print(f"Strides: {self.strides}")
        print("-" * 50)
    
    def get_2d_representation(self):
        """Convert flat data to 2D list representation."""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i * self.cols + j])
            result.append(row)
        return result
    
    def row_major_access(self, row, col):
        """
        Access element using row-major format formula.
        Formula: index = row * cols + col
        """
        if row >= self.rows or col >= self.cols or row < 0 or col < 0:
            raise IndexError(f"Index ({row}, {col}) out of bounds for shape {self.shape}")
        
        flat_index = row * self.cols + col
        value = self.data[flat_index]
        
        print(f"Row-major access A[{row}][{col}]:")
        print(f"  Formula: {row} * {self.cols} + {col} = {flat_index}")
        print(f"  data[{flat_index}] = {value}")
        return value
    
    def strides_access(self, row, col):
        """
        Access element using strides formula.
        Formula: index = row * stride[0] + col * stride[1]
        """
        if row >= self.rows or col >= self.cols or row < 0 or col < 0:
            raise IndexError(f"Index ({row}, {col}) out of bounds for shape {self.shape}")
        
        flat_index = row * self.strides[0] + col * self.strides[1]
        value = self.data[flat_index]
        
        print(f"Strides access A[{row}][{col}]:")
        print(f"  Formula: {row} * {self.strides[0]} + {col} * {self.strides[1]} = {flat_index}")
        print(f"  data[{flat_index}] = {value}")
        return value
    
    def verify_equivalence(self, row, col):
        """Verify that both methods give the same result."""
        val1 = self.row_major_access(row, col)
        print()
        val2 = self.strides_access(row, col)
        print(f"Both methods return: {val1} ✓" if val1 == val2 else "❌ Mismatch!")
        print("-" * 50)
        return val1 == val2
    
    def demonstrate_all_elements(self):
        """Show how to access all elements using both methods."""
        print("Demonstrating access to all elements:")
        print("2D representation:", self.get_2d_representation())
        print()
        
        for i in range(self.rows):
            for j in range(self.cols):
                print(f"Position ({i}, {j}):")
                self.verify_equivalence(i, j)
    
    def explain_strides(self):
        """Explain what strides mean in detail."""
        print("UNDERSTANDING STRIDES:")
        print(f"Shape: {self.shape} (rows={self.rows}, cols={self.cols})")
        print(f"Strides: {self.strides}")
        print()
        print("Stride meanings:")
        print(f"  strides[0] = {self.strides[0]} : To move 1 row down, skip {self.strides[0]} elements")
        print(f"  strides[1] = {self.strides[1]} : To move 1 col right, skip {self.strides[1]} element")
        print()
        print("Why these values?")
        print(f"  - Row stride = {self.cols} because each row has {self.cols} elements")
        print("  - Column stride = 1 because adjacent columns are 1 element apart")
        print("-" * 50)


def main():
    """Demonstrate the concepts with the given example."""
    
    # Example from the problem
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    shape = (2, 4)  # 2 rows, 4 columns
    
    print("ARRAY INDEXING CONCEPTS DEMONSTRATION")
    print("=" * 50)
    
    # Create the demo object
    demo = ArrayIndexingDemo(data, shape)
    
    # Show the 2D representation
    print("2D Array representation:")
    array_2d = demo.get_2d_representation()
    for i, row in enumerate(array_2d):
        print(f"  Row {i}: {row}")
    print()
    
    # Explain strides concept
    demo.explain_strides()
    
    # Test the specific example from the problem: A[1][2]
    print("TESTING SPECIFIC EXAMPLE: A[1][2]")
    demo.verify_equivalence(1, 2)
    
    # Show a few more examples
    print("MORE EXAMPLES:")
    demo.verify_equivalence(0, 0)  # First element
    demo.verify_equivalence(0, 3)  # End of first row
    demo.verify_equivalence(1, 0)  # Start of second row
    
    # Demonstrate all elements (optional - uncomment to see)
    # demo.demonstrate_all_elements()


# Additional utility functions
def create_array_from_shape(rows, cols, start_value=1):
    """Create a sequential array for testing."""
    data = list(range(start_value, start_value + rows * cols))
    return ArrayIndexingDemo(data, (rows, cols))


def demonstrate_different_shapes():
    """Show how the concept works with different array shapes."""
    print("\n" + "=" * 50)
    print("TESTING DIFFERENT SHAPES")
    print("=" * 50)
    
    # 3x3 array
    print("3x3 Array:")
    demo_3x3 = create_array_from_shape(3, 3)
    demo_3x3.verify_equivalence(2, 1)  # Test middle element
    
    # 1x8 array (row vector)
    print("1x8 Array (row vector):")
    demo_1x8 = ArrayIndexingDemo([1, 2, 3, 4, 5, 6, 7, 8], (1, 8))
    demo_1x8.verify_equivalence(0, 5)
    
    # 4x2 array
    print("4x2 Array:")
    demo_4x2 = ArrayIndexingDemo([1, 2, 3, 4, 5, 6, 7, 8], (4, 2))
    demo_4x2.verify_equivalence(3, 1)


if __name__ == "__main__":
    main()
    demonstrate_different_shapes()