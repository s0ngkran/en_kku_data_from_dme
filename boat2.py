import random




def pop_person(side):
    global boat
    random.shuffle(side)
    for i in side:
        if i < 0:
            side.remove(i)
            boat.append(i)
            return i
def pop_money(side):
    global boat
    random.shuffle(side)
    for i in side:
        if i > 0:
            side.remove(i)
            boat.append(i)
            return i
def pop_random(side):
    global boat
    random.shuffle(side)
    pop = side.pop(0)
    boat.append(pop)
    return pop

def ck(side):
    has_person = False
    for i in side:
        if i < 0:
            has_person = True
    if has_person:
        if sum(side) > 0:
            return False
    return True

def down_boat(side):
    global boat
    for i in boat:
        side.append(i)
    boat = []
def ck_sim_down(side):
    temp = []
    for i in boat:
        temp.append(i)
    for i in side:
        temp.append(i)
    return ck(temp)
def has_person(side):
    found = False
    for i in side:
        if i <0:
            found = True
            break
    return found
def print_():
    global boat, right, left
    print(left, boat, right)


def print_(side_name):
    global action, boat, right, left
    if side_name == 'left':
        ans = ''
        ans += convert_to_txt(left)
        ans += '|'
        ans += convert_boat(boat)
        ans += '->         |'
        ans += convert_to_txt(right)
    else:
        ans = ''
        ans += convert_to_txt(left)
        ans += '|        <-'
        ans += convert_boat(boat)
        ans += ' |'
        ans += convert_to_txt(right)
    action += ans + '\n'


my_map = {
    '-8': 'R ',
    '-5': 'G ',
    '-3': 'B ',
    '8': '8 ',
    '5': '5 ',
    '3': '3 ',
    'empty': '  '
}


def convert_boat(boat):
    temp = boat.copy()
    temp.sort()
    ans = '['
    for i in temp:
        ans += my_map[str(i)]
    if len(temp) == 1:
        ans += my_map['empty']
    ans += ']'
    return ans


def convert_to_txt(side):
    temp = side.copy()
    ans = ''

    temp.sort()
    for i in temp:
        ans += my_map[str(i)]
    for i in range(4-len(temp)):
        ans += my_map['empty']
    return ans

with open('boat.txt', 'w') as f:
    f.write('')

solution = []
n_solution = 1
while True:
    boat = []
    right = [8,5,3,-8,-5,-3]
    random.shuffle(right)
    left = []
    n_fail = 30
    action = ''
    try:
        # < 1
        while True:
            pop_person(right)
            pop_money(right)
            if ck(right) and ck(boat): 
                print_('right')
                down_boat(left)
                break
            else: 
                down_boat(right)

        # >
        pop_person(left)
        print_('left')
        down_boat(right)

        # <2
        _ = 0
        while True:
            _ += 1
            if _ > 20:
                1/0
            pop_person(right)
            pop_money(right)
            if ck(right) and ck(boat) and ck_sim_down(left): 
                print_('right')
                down_boat(left)
                break
            else: 
                down_boat(right)

        # >
        person = pop_person(left) 
        assert person == -8
        print_('left') 
        down_boat(right)

        # <3 # sure
        boat = [-5, -3] 
        right.remove(-5)
        right.remove(-3)
        print_('right')
        down_boat(left)

        # > # sure
        _ = 0
        while True:
            _ += 1
            if _ > n_fail:
                1/0
            pop_person(left)
            pop_money(left)
            if ck(left) and ck(boat) and ck_sim_down(right): 
                print_('left') 
                down_boat(right)
                break
            else: 
                down_boat(left)

        # <4
        _ = 0 # sure can be improve
        while True:
            _ += 1
            if _ > n_fail:
                1/0
            pop_person(right)
            pop_random(right)
            if ck(right) and ck(boat) and ck_sim_down(left): 
                print_('right')
                down_boat(left)
                break
            else: 
                down_boat(right)

        # > # sure that go back 2 things
        _ = 0
        while True:
            _ += 1
            if _ > n_fail:
                1/0
            pop_person(left)
            pop_random(left)
            if ck(left) and ck(boat) and ck_sim_down(right): 
                print_('left') 
                down_boat(right)
                break
            else: 
                down_boat(left)

        # <5 sure that go 2 things
        _ = 0
        while True:
            _ += 1
            if _ > n_fail:
                1/0
            pop_person(right)
            pop_random(right)
            if ck(right) and ck(boat) and ck_sim_down(left): 
                print_('right')
                down_boat(left)
                break
            else: 
                down_boat(right)

        # > # 
        _ = 0
        while True:
            _ += 1
            if _ > n_fail:
                1/0
            pop_person(left)
            if ck(left) and ck(boat) and ck_sim_down(right): 
                print_('left') 
                down_boat(right)
                break
            else: 
                down_boat(left)

        # <6 sure that go 2 things
        _ = 0
        while True:
            _ += 1
            if _ > n_fail:
                1/0
            pop_person(right)
            pop_random(right)
            if ck(right) and ck(boat) and ck_sim_down(left) :
                print_('right')
                down_boat(left)
                break
            else: 
                down_boat(right)

        # > # 
        _ = 0
        while True:
            _ += 1
            if _ > n_fail:
                1/0
            pop_person(left)
            if ck(left) and ck(boat) and ck_sim_down(right): 
                print_('left') 
                down_boat(right)
                break
            else: 
                down_boat(left)
        
        # <7 sure that go 2 things
        _ = 0
        while True:
            _ += 1
            if _ > n_fail:
                1/0
            pop_person(right)
            pop_random(right)
            if ck(right) and ck(boat) and ck_sim_down(left) :
                print_('right')
                down_boat(left)
                break
            else: 
                down_boat(right)
        if action not in solution:
            print('-',n_solution)
            solution.append(action)
            with open('boat.txt', 'a') as f:
                f.write('solution %d\n'%n_solution)
                f.write(action)
                f.write('\n')
            n_solution += 1
    except: 
        pass

