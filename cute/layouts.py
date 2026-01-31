import cutlass.cute as cute
import cutlass

@cute.jit
def printdemo(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    print(a)
    print(b)
    cute.printf("Dynamic a: {}\n", a)
    cute.printf("Dynamic b: {}\n", b)
    layout = cute.make_layout((a, b), stride=(1,4))
    print("static layout:", layout)
    cute.printf("Dynamic layout: {}\n", layout)

printdemo(cutlass.Int32(8), 2)