import numpy as np
import sys

TOL = 1e-4

def run_checks():
    # Load actual outputs
    try:
        Q = np.genfromtxt('./outputs/demo_run.7.Q')
        P = np.genfromtxt('./outputs/demo_run.7.P')
    except FileNotFoundError as e:
        print('Could not find output files. Please make sure to run the demo before running the test.')
        return False
    except Exception as e:
        raise e
    # Load expected outputs
    try:
        expected_Q = np.genfromtxt('./outputs/demo_run.7.Q.expected')
        expected_P = np.genfromtxt('./outputs/demo_run.7.P.expected')
    except FileNotFoundError as e:
        print('Could not expected output files. Please make sure they are present in the outputs folder of the demo.')
        return False
    except Exception as e:
        raise e
    # Check mean error is below tolerance
    return np.allclose(Q, expected_Q) and np.allclose(P, expected_P)

if __name__ == '__main__':
    passed = run_checks()
    print(f'Output and expected output are {"" if passed else "NOT "}similar.')
    sys.exit(0)
