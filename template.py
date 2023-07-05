import sys
# sys.setrecursionlimit(10**6)

def iterate_tokens():
    for line in sys.stdin:
        for word in line.split():
            yield word
tokens = iterate_tokens()

def read() -> str:
    return next(tokens)

def read_int() -> int:
    return int(next(tokens))

def read_float() -> float:
    return float(next(tokens))

def main():
    return

if __name__ == '__main__':
    main()
