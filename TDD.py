def test_is_prime():
    try:
        assert (is_prime(5) == True)
    except AssertionError:
        print('Test 1 failed , 5 is a prime check test 1')
    if is_prime(5)==False:
        print("test failed")


def is_prime(n):
    for i in range(1,int(n**0.5+1)):
        if n%i == 0:
            return False
    return True

if __name__ == '__main__':
    test_is_prime()