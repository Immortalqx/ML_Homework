if __name__ == '__main__':
    line = input()
    number_list = []
    number_list.extend([int(item) for item in line.split(",")])
    number_list.sort(reverse=True)
    print(number_list)
