import sys, importlib

REQUIRED = ["numpy", "torch", "transformers"]

def check(m):
    try:
        importlib.import_module(m)
        return True
    except Exception as e:
        print(f" - {m}: FAIL -> {e}")
        return False

def main():
    print("Python:", sys.version.split()[0])
    ok = True
    for m in REQUIRED:
        if check(m):
            print(f" - {m}: OK")
        else:
            ok = False
    print("\nResult:", "✅ PASS" if ok else "❌ FAIL")
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
