from math import sqrt
def fitting(input1:str):
    input_list = input1.split()
    input_int = []
    for i in input_list:
        input_int.append(int(i))
    height_of_locker = input_int[0]
    length_of_locker = input_int[1]
    width_of_locker = input_int[2]
    length_of_magic_stick = input_int[3]
    length_of_diagonal = sqrt(height_of_locker**2 + length_of_locker**2 + width_of_locker**2 )
    # if height_of_locker >= length_of_magic_stick:
    #     return f"stick fits vertically"
    # elif length_of_locker >= length_of_magic_stick:
    #     return f"stick fits horizontally"
    # elif length_of_diagonal>= length_of_magic_stick:
    #     return f"stick fits diagonally"
    # else:
    #     pass

    if height_of_locker >= length_of_magic_stick and length_of_locker >= length_of_magic_stick and length_of_diagonal>= length_of_magic_stick:
        return f"Fits three directions"
    if (height_of_locker >= length_of_magic_stick and length_of_locker >= length_of_magic_stick) or (height_of_locker >= length_of_magic_stick and length_of_diagonal>= length_of_magic_stick) or (length_of_locker >= length_of_magic_stick and length_of_diagonal>= length_of_magic_stick):
        return f"Fits two directions"
    if height_of_locker >= length_of_magic_stick or length_of_locker >= length_of_magic_stick or length_of_diagonal>= length_of_magic_stick:
        return f"Fits one direction"

fitting(input1=f"1 2 3 4")
