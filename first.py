for N in range(999, 99, -1):
    N_list = [int(i) for i in str(N)]
    s = []
    for i in range(0, 2):
        s.append(N_list[i] + N_list[i+1])
    if str(max(s)) + str(min(s)) == "1412":
        print(N)
